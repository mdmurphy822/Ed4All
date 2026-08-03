"""Stage-2 wiring for the post-hoc citation re-selection pass.

Drives ``_run_stage2_window_synthesis`` (chunk_window mode) end-to-end with a
mock provider that deliberately cites the same-chapter NEIGHBOR chunk (the
measured 7B sloppiness) and asserts the ``ED4ALL_OBJECTIVE_CITATION_RESELECT``
pass re-cites the best supporter from the widened window ∪ chapter pool BEFORE
CO-id minting / TO derivation — and that the flag-off path is byte-identical,
plus the cross-window same-chapter hole FIX 1 closes. Hermetic: FakeEmbed
(token-hash unit vectors), passthrough grounding (no NLI load), no network.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402


class _ProviderError(RuntimeError):
    pass


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


class _SloppyCiterProvider:
    """Cites the FIRST chunk of each window (the intro neighbor) while the
    authored statement matches the SECOND chunk's vocabulary."""

    _model = "test-model-v1"
    _max_tokens = 0

    def __init__(self) -> None:
        self.window_calls: List[str] = []

    def batch_chapters(self, items, batch_size: int = 10):
        return [
            items[i:i + batch_size] for i in range(0, len(items), batch_size)
        ]

    def synthesize_window_objectives(
        self, window, *, course_name, draft_terminal_objectives
    ):
        cid = str(window.get("chapter_id"))
        self.window_calls.append(cid)
        chunk_ids = [str(c) for c in window.get("chunk_ids") or []]
        stmt = {
            "ch1": "Solve linear equations using the substitution method.",
            "ch2": "Explain photosynthesis and the role of the chloroplast.",
        }.get(cid, f"Objective for {cid}.")
        return {
            "candidate_objectives": [{
                "statement": stmt,
                # Sloppy: cite the window's FIRST chunk (the intro neighbor).
                "source_chunk_ids": chunk_ids[:1],
                "source_refs": [{"ref": cid, "chunk_ids": chunk_ids[:1]}],
            }]
        }

    def author_terminal_for_cluster(
        self, cluster_cos, *, course_name, cluster_index
    ):
        rep = max(
            cluster_cos, key=lambda c: len(str(c.get("statement") or ""))
        )
        return {
            "statement": f"TO summarising: {rep.get('statement')}",
            "bloom_level": "understand",
            "source_refs": [],
        }

    def reconcile_terminal_objectives(self, *a, **kw):
        return {"terminal_objectives": [
            {"id": "TO-01", "statement": "Legacy TO."}
        ]}


class _CrossWindowProvider:
    """Authors, for each 1-chunk window, a statement matching a DIFFERENT
    same-chapter window's chunk while citing its OWN chunk — reproducing the
    FIX 1 hole: the true supporter is in the CO's chapter but NOT its window."""

    _model = "test-model-v1"
    _max_tokens = 0

    def batch_chapters(self, items, batch_size: int = 10):
        return [
            items[i:i + batch_size] for i in range(0, len(items), batch_size)
        ]

    def synthesize_window_objectives(
        self, window, *, course_name, draft_terminal_objectives
    ):
        chunk_ids = [str(c) for c in window.get("chunk_ids") or []]
        own = chunk_ids[0] if chunk_ids else ""
        # Statement vocabulary points at the OTHER chunk in the same chapter.
        stmt = {
            "c1": "Solve linear equations using the substitution method.",
            "c2": "Welcome and course introduction overview reading notes.",
        }.get(own, f"Objective for {own}.")
        return {
            "candidate_objectives": [{
                "statement": stmt,
                "source_chunk_ids": [own],           # cite own (wrong) window
                "source_refs": [{"ref": str(window.get("chapter_id")),
                                 "chunk_ids": [own]}],
            }]
        }

    def author_terminal_for_cluster(
        self, cluster_cos, *, course_name, cluster_index
    ):
        rep = max(
            cluster_cos, key=lambda c: len(str(c.get("statement") or ""))
        )
        return {
            "statement": f"TO summarising: {rep.get('statement')}",
            "bloom_level": "understand",
            "source_refs": [],
        }

    def reconcile_terminal_objectives(self, *a, **kw):
        return {"terminal_objectives": [
            {"id": "TO-01", "statement": "Legacy TO."}
        ]}


class _FabricatedCiterProvider:
    """Cites a WHOLLY FABRICATED id (a descriptive topic-label that resolves
    against NOTHING in the real chunkset) — the measured nano-omni failure that
    leaves objectives with ~0 real grounding. The authored statement matches the
    window's SECOND real chunk's vocabulary."""

    _model = "test-model-v1"
    _max_tokens = 0

    def batch_chapters(self, items, batch_size: int = 10):
        return [
            items[i:i + batch_size] for i in range(0, len(items), batch_size)
        ]

    def synthesize_window_objectives(
        self, window, *, course_name, draft_terminal_objectives
    ):
        cid = str(window.get("chapter_id"))
        stmt, label = {
            "ch1": (
                "Solve linear equations using the substitution method.",
                "Solve Linear Equations",
            ),
            "ch2": (
                "Explain photosynthesis and the role of the chloroplast.",
                "Explain Photosynthesis",
            ),
        }.get(cid, (f"Objective for {cid}.", "Fabricated Topic"))
        return {
            "candidate_objectives": [{
                "statement": stmt,
                # Fabricated: a topic-label, NOT any real window chunk id.
                "source_chunk_ids": [label],
                "source_refs": [{"ref": cid, "chunk_ids": [label]}],
            }]
        }

    def author_terminal_for_cluster(
        self, cluster_cos, *, course_name, cluster_index
    ):
        rep = max(
            cluster_cos, key=lambda c: len(str(c.get("statement") or ""))
        )
        return {
            "statement": f"TO summarising: {rep.get('statement')}",
            "bloom_level": "understand",
            "source_refs": [],
        }

    def reconcile_terminal_objectives(self, *a, **kw):
        return {"terminal_objectives": [
            {"id": "TO-01", "statement": "Legacy TO."}
        ]}


class _PassthroughGrounding:
    def __init__(self, grounded: List[Dict[str, Any]]) -> None:
        self.grounded = grounded
        self.ungrounded: List[Dict[str, Any]] = []
        self.available = False
        self.dropped_count = 0
        self.reground_count = 0


def _chunks() -> List[Dict[str, Any]]:
    # Two chapters x two chunks; chunk 1 of each chapter is a low-relevance
    # intro, chunk 2 carries the statement vocabulary.
    return [
        {"id": "c1", "text": "welcome chapter introduction overview reading"},
        {"id": "c2", "text": (
            "solve linear equations substitution method substitute solve "
            "equations method"
        )},
        {"id": "c3", "text": "chapter opening notes preview outline reading"},
        {"id": "c4", "text": (
            "photosynthesis chloroplast light energy explain photosynthesis "
            "chloroplast"
        )},
    ]


def _run(monkeypatch, *, reselect: bool):
    import lib.embedding.providers as _providers
    import lib.objectives.objective_grounding as _og

    monkeypatch.setattr(
        _providers, "build_embedding_client", lambda **_k: FakeEmbed(),
    )
    monkeypatch.setattr(
        _og,
        "ground_candidates",
        lambda cands, chunks_by_id, require=False: _PassthroughGrounding(
            list(cands)
        ),
    )
    # Big window → each chapter is ONE 2-chunk window.
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", "100000")
    if reselect:
        monkeypatch.setenv("ED4ALL_OBJECTIVE_CITATION_RESELECT", "1")
    else:
        monkeypatch.delenv(
            "ED4ALL_OBJECTIVE_CITATION_RESELECT", raising=False,
        )

    all_chunks = _chunks()
    chunks_by_id = {c["id"]: c for c in all_chunks}
    chapters = [
        {"id": "ch1", "chapter_text": "x" * 10},
        {"id": "ch2", "chapter_text": "y" * 10},
    ]
    provider = _SloppyCiterProvider()
    out = asyncio.run(pt._run_stage2_window_synthesis(
        provider=provider,
        provider_error=_ProviderError,
        chapters=chapters,
        draft_tos=[{"statement": "Draft course objective."}],
        chunks_by_id=chunks_by_id,
        all_chunks=all_chunks,
        grounding_mode="chunk_window",
        course_name="FXMATH_101",
        provider_env="local",
        chapter_synthesis_failures=[],
        mint_lo_id=_mint,
        kwargs={},
        capture=None,
        checkpoint_path=None,
    ))
    return out


def _run_fabricated(monkeypatch, *, reselect: bool):
    """Same harness as ``_run`` but with the fabricated-citer provider."""
    import lib.embedding.providers as _providers
    import lib.objectives.objective_grounding as _og

    monkeypatch.setattr(
        _providers, "build_embedding_client", lambda **_k: FakeEmbed(),
    )
    monkeypatch.setattr(
        _og,
        "ground_candidates",
        lambda cands, chunks_by_id, require=False: _PassthroughGrounding(
            list(cands)
        ),
    )
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", "100000")
    if reselect:
        monkeypatch.setenv("ED4ALL_OBJECTIVE_CITATION_RESELECT", "1")
    else:
        monkeypatch.delenv("ED4ALL_OBJECTIVE_CITATION_RESELECT", raising=False)

    all_chunks = _chunks()
    chunks_by_id = {c["id"]: c for c in all_chunks}
    chapters = [
        {"id": "ch1", "chapter_text": "x" * 10},
        {"id": "ch2", "chapter_text": "y" * 10},
    ]
    out = asyncio.run(pt._run_stage2_window_synthesis(
        provider=_FabricatedCiterProvider(),
        provider_error=_ProviderError,
        chapters=chapters,
        draft_tos=[{"statement": "Draft course objective."}],
        chunks_by_id=chunks_by_id,
        all_chunks=all_chunks,
        grounding_mode="chunk_window",
        course_name="FXMATH_101",
        provider_env="local",
        chapter_synthesis_failures=[],
        mint_lo_id=_mint,
        kwargs={},
        capture=None,
        checkpoint_path=None,
    ))
    return out


def test_reselect_grounds_wholly_fabricated_citations(monkeypatch):
    """REAL-grounding fix: a CO whose ONLY cited id is a fabricated topic-label
    is re-grounded to the cosine-best REAL chunk from its window pool — not left
    at ~0 real grounding."""
    chapter_cos, terminal, _mm, signals = _run_fabricated(
        monkeypatch, reselect=True
    )
    assert len(chapter_cos) == 2
    solve_co = _co_by_statement(chapter_cos, "substitution")
    photo_co = _co_by_statement(chapter_cos, "photosynthesis")
    # The fabricated label is gone; a REAL cosine-relevant chunk is cited.
    assert solve_co["source_chunk_ids"] == ["c2"]
    assert photo_co["source_chunk_ids"] == ["c4"]
    assert "Solve Linear Equations" not in solve_co["source_chunk_ids"]
    assert solve_co["source_refs"][0]["chunk_ids"] == ["c2"]
    # Grounding is now > 0 and the fabricated ids counted as pool misses.
    assert signals["citations_reselected"] == 2
    assert signals["citation_density_after"] >= 2
    assert signals["reselect_pool_misses"] >= 2  # both fabricated labels dropped
    # The transient Pass-B provenance stamp never leaks onto the objectives.
    assert all("_reselect_window_pool" not in co for co in chapter_cos)


def test_reselect_off_keeps_fabricated_and_leaks_no_stamp(monkeypatch):
    """Flag OFF → byte-identical: fabricated citations survive verbatim and no
    ``_reselect_window_pool`` provenance stamp is ever added."""
    chapter_cos, _t, _mm, signals = _run_fabricated(monkeypatch, reselect=False)
    solve_co = _co_by_statement(chapter_cos, "substitution")
    photo_co = _co_by_statement(chapter_cos, "photosynthesis")
    assert solve_co["source_chunk_ids"] == ["Solve Linear Equations"]
    assert photo_co["source_chunk_ids"] == ["Explain Photosynthesis"]
    assert signals["citations_reselected"] == 0
    assert all("_reselect_window_pool" not in co for co in chapter_cos)


def _run_cross_window(monkeypatch):
    """Drive the hook with 1-chunk windows so a chapter spans TWO windows —
    the FIX 1 hole: cited chunk resolves to a window WITHOUT the supporter."""
    import lib.embedding.providers as _providers
    import lib.objectives.objective_grounding as _og

    monkeypatch.setattr(
        _providers, "build_embedding_client", lambda **_k: FakeEmbed(),
    )
    monkeypatch.setattr(
        _og, "ground_candidates",
        lambda cands, chunks_by_id, require=False: _PassthroughGrounding(
            list(cands)
        ),
    )
    # Tiny window budget → one chunk per window → chapter ch1 spans 2 windows.
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", "1")
    monkeypatch.setenv("ED4ALL_OBJECTIVE_CITATION_RESELECT", "1")

    all_chunks = [
        {"id": "c1", "text": "welcome chapter introduction overview reading"},
        {"id": "c2", "text": (
            "solve linear equations substitution method substitute solve "
            "equations method"
        )},
    ]
    chunks_by_id = {c["id"]: c for c in all_chunks}
    chapters = [{"id": "ch1", "chapter_text": "x" * 10}]
    out = asyncio.run(pt._run_stage2_window_synthesis(
        provider=_CrossWindowProvider(),
        provider_error=_ProviderError,
        chapters=chapters,
        draft_tos=[{"statement": "Draft course objective."}],
        chunks_by_id=chunks_by_id,
        all_chunks=all_chunks,
        grounding_mode="chunk_window",
        course_name="FXMATH_101",
        provider_env="local",
        chapter_synthesis_failures=[],
        mint_lo_id=_mint,
        kwargs={},
        capture=None,
        checkpoint_path=None,
    ))
    return out


def test_reselect_reaches_cross_window_chapter_supporter(monkeypatch):
    chapter_cos, _t, _mm, signals = _run_cross_window(monkeypatch)
    solve_co = _co_by_statement(chapter_cos, "substitution")
    assert solve_co is not None
    # The supporter (c2) lives in a DIFFERENT window than the cited c1, but the
    # SAME chapter (ch1). Only the widened window ∪ chapter pool can reach it.
    assert solve_co["source_chunk_ids"] == ["c2"]
    # source_refs derived from the kept chunk's (annotated) chapter_id.
    assert solve_co["source_refs"] == [{"ref": "ch1", "chunk_ids": ["c2"]}]
    assert signals["citations_reselected"] >= 1


def _co_by_statement(cos, needle: str) -> Optional[Dict[str, Any]]:
    for co in cos:
        if needle in str(co.get("statement") or ""):
            return co
    return None


def test_reselect_on_corrects_neighbor_citation(monkeypatch):
    chapter_cos, terminal, _mm, signals = _run(monkeypatch, reselect=True)
    assert len(chapter_cos) == 2 and terminal
    solve_co = _co_by_statement(chapter_cos, "substitution")
    photo_co = _co_by_statement(chapter_cos, "photosynthesis")
    # Neighbor miss corrected: the strongest in-window supporter is cited.
    assert solve_co["source_chunk_ids"][0] == "c2"
    assert photo_co["source_chunk_ids"][0] == "c4"
    # source_refs mirrored.
    assert solve_co["source_refs"][0]["chunk_ids"] == (
        solve_co["source_chunk_ids"]
    )
    # Signals surfaced on grounding_signals (ride objective_grounding_filter).
    assert signals["citations_reselected"] == 2
    assert signals["citation_density_before"] == 2   # c1 + c3
    assert signals["citation_density_after"] >= 2    # c2 + c4 (+ maybe more)
    assert "reselect_pool_misses" in signals
    # Downstream continuity: TO source union reflects the NEW citations
    # (the pass ran BEFORE TO derivation).
    union: set = set()
    for to in terminal:
        for ref in to.get("source_refs") or []:
            union.update(ref.get("chunk_ids") or [])
    if union:  # TO child-annotation ran
        assert "c2" in union or "c4" in union


def test_reselect_off_is_byte_identical(monkeypatch):
    chapter_cos, _t, _mm, signals = _run(monkeypatch, reselect=False)
    solve_co = _co_by_statement(chapter_cos, "substitution")
    photo_co = _co_by_statement(chapter_cos, "photosynthesis")
    # The sloppy neighbor citations survive verbatim when the flag is off.
    assert solve_co["source_chunk_ids"] == ["c1"]
    assert photo_co["source_chunk_ids"] == ["c3"]
    assert signals["citations_reselected"] == 0
    assert signals["citation_density_before"] == 0  # pass never ran
    assert signals["citation_density_after"] == 0


def test_synthesis_signals_carry_all_reselect_counters(monkeypatch):
    """The grounding_signals dict returned by ``_run_stage2_window_synthesis``
    (the 4th tuple element that the ``plan_course_structure`` TOOL return
    surfaces under ``grounding_signals``) carries EVERY reselect counter the
    audit reads — so wiring the whole dict into the tool return exposes them
    all, not just the three spot-checked above."""
    _cc, _t, _mm, signals = _run(monkeypatch, reselect=True)
    for key in (
        "citations_reselected",
        "citation_density_before",
        "citation_density_after",
        "reselect_pool_misses",
    ):
        assert key in signals, f"reselect counter {key} missing from signals"
        assert isinstance(signals[key], int)


def test_plan_course_structure_tool_return_surfaces_grounding_signals(
    tmp_path, monkeypatch
):
    """End-to-end guard: the ``plan_course_structure`` TOOL return exposes a
    ``grounding_signals`` dict, so a post-run audit reads reselect counters
    from ``phase_outputs.course_planning.grounding_signals`` (persisted via the
    workflows.yaml ``outputs`` whitelist + executor json.loads) instead of
    scraping INFO logs.

    Drives the full tool via the registry on the user-supplied-objectives path
    (Stage-2 window synthesis does NOT run there, so ``grounding_signals`` is an
    empty ``{}`` — the KEY is nonetheless always present, per the additive
    contract). This test FAILS if someone drops the
    ``"grounding_signals": _grounding_signals`` wiring from the tool return.
    """
    import json

    fake_root = tmp_path / "root"
    fake_root.mkdir()
    exports = fake_root / "Courseforge" / "exports"
    exports.mkdir(parents=True)
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)
    monkeypatch.setattr(pt, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(pt, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pt,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    project_id = "PROJ-RS_101-20260701000000"
    project_dir = exports / project_id
    project_dir.mkdir()
    for subdir in (
        "00_template_analysis", "01_learning_objectives",
        "02_course_planning", "03_content_development",
        "04_quality_validation", "05_final_package",
    ):
        (project_dir / subdir).mkdir()
    (project_dir / "project_config.json").write_text(
        json.dumps({
            "project_id": project_id,
            "course_name": "RS_101",
            "duration_weeks": 6,
            "credit_hours": 3,
        }, indent=2),
        encoding="utf-8",
    )
    supplied = project_dir / "supplied_objectives.json"
    supplied.write_text(json.dumps({
        "duration_weeks": 6,
        "terminal_objectives": [
            {"id": f"TO-{i:02d}", "statement": f"Terminal outcome {i}.",
             "bloom_level": "analyze"}
            for i in range(1, 4)
        ],
        "chapter_objectives": [{
            "chapter": "Week 1",
            "objectives": [
                {"id": f"CO-{i:02d}", "statement": f"Chapter outcome {i}.",
                 "bloom_level": "remember"}
                for i in range(1, 13)
            ],
        }],
    }), encoding="utf-8")

    registry = pt._build_tool_registry()
    fn = registry["plan_course_structure"]
    result = json.loads(asyncio.run(fn(
        project_id=project_id,
        objectives_path=str(supplied),
        duration_weeks=6,
        duration_weeks_explicit=False,
    )))
    assert result["success"]
    # The additive contract: the key is ALWAYS present (empty {} here because
    # Stage-2 window synthesis never ran on the user-supplied path).
    assert "grounding_signals" in result, (
        "plan_course_structure tool return dropped the grounding_signals key"
    )
    assert isinstance(result["grounding_signals"], dict)
    # JSON round-tripped cleanly (no dataclasses / Paths leaked into the dict).
    assert result["grounding_signals"] == {}
