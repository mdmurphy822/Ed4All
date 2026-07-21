"""W2 §5.5 / §5.6 — end-to-end Pass B→C→D→E orchestration (hermetic).

Drives ``MCP.tools.pipeline_tools._run_stage2_window_synthesis`` (the W2 core
that ``_plan_course_structure`` calls) with a stub TextbookSynthesisProvider,
fake NLI, and fake embeddings — no live model. Asserts the four headline
invariants over the produced CO set:

1. every CO carries non-empty grounded ``source_chunk_ids`` ⊆ the fixture
   chunkset ids;
2. no CO statement is from the ungrounded set (the fake NLI's low-scored ones);
3. near-duplicate candidates collapsed (CO count < raw candidate count);
4. every CO gets a ``terminal_id`` resolving to an emitted TO (backlink);
plus the chunk-window vs chapter-fallback grounding modes.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
import lib.objectives.objective_grounding as og  # noqa: E402
import lib.embedding.providers as ep  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed, FakeNli  # noqa: E402
from Courseforge.generators._textbook_synthesis_provider import (  # noqa: E402
    TextbookSynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Stub provider — window + chapter dispatch, reconcile, batch splitter.
# ---------------------------------------------------------------------------
class _StubProvider:
    """Emits canned candidate objectives per window / chapter + a reconcile."""

    _max_tokens = 4096

    def __init__(self, *, mode: str = "window") -> None:
        self._mode = mode

    def synthesize_window_objectives(self, window, *, course_name,
                                     draft_terminal_objectives=None):
        chunk_ids = list(window.get("chunk_ids") or [])
        cid = window.get("chapter_id")
        # Two near-duplicate "prime" candidates across this window's chunks +
        # one disjoint "triangle" candidate + one UNGROUNDED candidate whose
        # statement shares nothing with its cited chunk text.
        cands = [
            {
                "statement": "Define a prime number precisely.",
                "bloom_level": "understand",
                "source_chunk_ids": chunk_ids[:1] or chunk_ids,
                "grounded_citation": True,
                "source_refs": [{"ref": cid, "chunk_ids": chunk_ids[:1]}],
                "chapter_id": cid,
                "sub_objectives": [],
            },
            {
                "statement": "Define what a prime number is precisely.",
                "bloom_level": "understand",
                "source_chunk_ids": chunk_ids[:1] or chunk_ids,
                "grounded_citation": True,
                "source_refs": [{"ref": cid, "chunk_ids": chunk_ids[:1]}],
                "chapter_id": cid,
                "sub_objectives": [],
            },
            {
                "statement": "Compute triangle area from base and height.",
                "bloom_level": "apply",
                "source_chunk_ids": chunk_ids,
                "grounded_citation": True,
                "source_refs": [{"ref": cid, "chunk_ids": chunk_ids}],
                "chapter_id": cid,
                "sub_objectives": [],
            },
            {
                "statement": "Recite medieval heraldry banner colours.",
                "bloom_level": "remember",
                "source_chunk_ids": chunk_ids[:1] or chunk_ids,
                "grounded_citation": True,
                "source_refs": [{"ref": cid, "chunk_ids": chunk_ids[:1]}],
                "chapter_id": cid,
                "sub_objectives": [],
            },
        ]
        return {"chapter_id": cid, "window_index": window.get("window_index"),
                "candidate_objectives": cands}

    def synthesize_chapter_objectives(self, chapter, *, course_name,
                                      draft_terminal_objectives=None):
        cid = chapter.get("id")
        return {
            "chapter_id": cid,
            "chapter_objectives": [
                {
                    "statement": "Define a prime number precisely.",
                    "bloom_level": "understand",
                    "source_refs": [{"ref": cid, "chunk_ids": []}],
                    "chapter_id": cid,
                    "sub_objectives": [],
                },
                {
                    "statement": "Compute triangle area from base and height.",
                    "bloom_level": "apply",
                    "source_refs": [{"ref": cid, "chunk_ids": []}],
                    "chapter_id": cid,
                    "sub_objectives": [],
                },
            ],
        }

    def reconcile_terminal_objectives(self, draft_tos, chapter_cos, *,
                                      course_name):
        return {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Understand number theory primes."},
                {"id": "TO-02", "statement": "Apply geometry triangle area."},
            ]
        }

    def author_terminal_for_cluster(self, cluster_cos, *, course_name,
                                    cluster_index):
        # WS1 — author ONE TO per cluster (no id; caller mints). The longest
        # member statement seeds the summary so the output is deterministic.
        rep = max(
            cluster_cos, key=lambda c: len(str(c.get("statement") or "")),
            default={},
        )
        return {
            "statement": f"Terminal: {rep.get('statement') or ''}",
            "bloom_level": "understand",
            "source_refs": [],
        }

    @staticmethod
    def batch_chapters(items, batch_size: int = 10):
        size = max(1, int(batch_size))
        return [list(items[i:i + size]) for i in range(0, len(items), size)]


def _chunk(cid, slug, body):
    return {
        "id": cid,
        "text": body,
        "source": {"source_references": [{"sourceId": f"semantik:{slug}#{cid}",
                                           "role": "primary"}]},
    }


def _chapter(cid, slug):
    return {"id": cid, "source_file": f"{slug}.html",
            "chapter_text": "chapter prose " * 20}


def _patch_nli_and_embed(monkeypatch):
    """Bind a fake NLI into the grounding scorer + a fake embed everywhere."""
    fake_nli = FakeNli()
    real_score = og.score_groundedness

    def _scored(answer_text, cited_passages, **kw):
        # ground_candidates passes nli=None explicitly; force the fake when the
        # caller did not supply a concrete classifier (mimics the singleton
        # being present in a CI box that has the NLI extras).
        if kw.get("nli") is None:
            kw["nli"] = fake_nli
        return real_score(answer_text, cited_passages, **kw)

    monkeypatch.setattr(og, "score_groundedness", _scored)
    monkeypatch.setattr(ep, "build_embedding_client",
                        lambda *a, **k: FakeEmbed())
    monkeypatch.setattr(ep, "allow_fake_enabled", lambda: True)


def _run_window(monkeypatch, mode="window"):
    _patch_nli_and_embed(monkeypatch)
    chapters = [_chapter("ch1", "one"), _chapter("ch2", "two")]
    if mode == "window":
        all_chunks = [
            _chunk("c1", "one", "prime number integer divisor factor define"),
            _chunk("c2", "one", "triangle area base height compute geometry"),
            _chunk("c3", "two", "prime number integer divisor factor define"),
        ]
    else:
        all_chunks = []
    chunks_by_id = {c["id"]: c for c in all_chunks}
    draft_tos = [{"statement": "Understand primes and triangle area."}]
    failures: List[str] = []

    def _mint(level, idx):
        return f"CO-{idx:02d}"

    return asyncio.run(
        pt._run_stage2_window_synthesis(
            provider=_StubProvider(mode=mode),
            provider_error=TextbookSynthesisProviderError,
            chapters=chapters,
            draft_tos=draft_tos,
            chunks_by_id=chunks_by_id,
            all_chunks=all_chunks,
            grounding_mode=("chunk_window" if mode == "window"
                            else "chapter_fallback"),
            course_name="MATH_101",
            provider_env="local",
            chapter_synthesis_failures=failures,
            mint_lo_id=_mint,
            kwargs={},
        )
    )


def test_window_mode_grounded_deduped_backlinked(monkeypatch):
    chapter_cos, terminal, mint_method, signals = _run_window(
        monkeypatch, mode="window"
    )
    assert chapter_cos, "expected canonical COs"
    # 1. every CO carries non-empty source_chunk_ids ⊆ the fixture chunk ids.
    fixture_ids = {"c1", "c2", "c3"}
    for co in chapter_cos:
        ids = co.get("source_chunk_ids") or []
        assert ids, f"CO {co.get('id')} has empty source_chunk_ids"
        assert set(ids) <= fixture_ids
    # 2. no CO statement is from the ungrounded set (heraldry).
    statements = {co["statement"] for co in chapter_cos}
    assert not any("heraldry" in s for s in statements)
    # 3. near-duplicate prime candidates collapsed → CO count < raw candidates.
    assert signals["candidate_count"] > signals["canonical_co_count"]
    assert signals["near_dup_pairs"] >= 1
    # 4. every CO has a terminal_id resolving to an emitted TO.
    to_ids = {t["id"] for t in terminal}
    for co in chapter_cos:
        assert co.get("terminal_id") in to_ids
    # signals sanity.
    assert signals["grounding_mode"] == "chunk_window"
    assert signals["ungrounded_dropped"] >= 1
    assert signals["nli_available"] is True
    assert mint_method.startswith("textbook_synthesis_window:")


def test_chapter_fallback_keeps_all_no_drop(monkeypatch):
    """chunkset missing → chapter_fallback: empty chunk_ids, nothing dropped."""
    chapter_cos, terminal, mint_method, signals = _run_window(
        monkeypatch, mode="chapter"
    )
    assert chapter_cos
    assert signals["grounding_mode"] == "chapter_fallback"
    # §5.6 — fallback drops nothing on grounding (no chunks to ground against).
    assert signals["ungrounded_dropped"] == 0
    # Every CO still gets a backlinked terminal_id.
    to_ids = {t["id"] for t in terminal}
    for co in chapter_cos:
        assert co.get("terminal_id") in to_ids


# ---------------------------------------------------------------------------
# Decision-capture contract — both new decision types validate under strict.
# ---------------------------------------------------------------------------
def test_new_decision_types_fire_under_strict(monkeypatch, tmp_path):
    """objective_grounding_filter + block_objective_alignment round-trip a
    real DecisionCapture with DECISION_VALIDATION_STRICT=true (the enum-add
    hard-ordering prerequisite)."""
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path))
    from lib.decision_capture import DecisionCapture

    cap = DecisionCapture(course_code="MATH_101", phase="course-outliner",
                          tool="courseforge", streaming=False)
    # Must NOT raise under strict validation.
    cap.log_decision(
        decision_type="objective_grounding_filter",
        decision="objective_grounding_filter:MATH_101:3/5 grounded",
        rationale=("course=MATH_101; grounding_mode=chunk_window; "
                   "candidate_count=5; grounded_count=3; ungrounded_dropped=1"),
    )
    cap.log_decision(
        decision_type="block_objective_alignment",
        decision="block_objective_alignment:b1:aligned",
        rationale=("block_id=b1; block_type=concept; existing_refs=0; "
                   "added_refs=1; final_refs=1; threshold=0.55"),
    )
    types = {d.get("decision_type") for d in cap.decisions}
    assert "objective_grounding_filter" in types
    assert "block_objective_alignment" in types
