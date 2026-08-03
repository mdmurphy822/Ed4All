"""Defect-A chapter-anchored TO derivation — Stage-2 integration invariants.

Two altitudes, both hermetic (no GPU / ollama / network / NLI):

1. Direct-call tests on ``_derive_terminals_chapter_anchored`` (the module-loop
   analogue of the bottom-up cluster tests in
   ``MCP/tools/tests/test_pipeline_tools_stage2_stop.py``): the ``chapter_anchored``
   signal, TO order == book (module) order, backlink no-op on the pre-set
   ``terminal_id``, per-module sidecar reuse, and — the graceful-stop contract —
   a stop armed mid-derivation pauses at the SAME ``stage2_clusters`` boundary
   with ``sidecar records == author calls`` (mirrors the bottom-up stop test).

2. One call-site test through ``_run_stage2_window_synthesis`` proving the
   selection wiring: flag ON (+ chunk_window grounding) → ``to_derivation ==
   "chapter_anchored"``; flag OFF → byte-identical bottom-up path
   (``to_derivation == "bottom_up"``), untouched.

Synthetic ``mod-a`` / ``mod-b`` / ``mod-c`` modules only — no corpus vocabulary,
no course slugs, no data paths.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402

_RUN_ID = "STOP_CHAPTER_ANCHOR_TESTRUN"


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


class _ProviderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Synthetic 3-module corpus
# ---------------------------------------------------------------------------
def _chunk(cid: str, module_id: str, *, pos: int = 0) -> Dict[str, Any]:
    return {
        "id": cid,
        "text": f"Body of {cid}.",
        "source": {
            "module_id": module_id,
            "module_title": module_id.upper(),
            "position_in_module": pos,
        },
    }


def _three_module_all_chunks() -> List[Dict[str, Any]]:
    # Book order a → b → c (first-occurrence order).
    return [
        _chunk("a1", "mod-a", pos=0),
        _chunk("a2", "mod-a", pos=1),
        _chunk("b1", "mod-b", pos=0),
        _chunk("b2", "mod-b", pos=1),
        _chunk("c1", "mod-c", pos=0),
        _chunk("c2", "mod-c", pos=1),
    ]


def _cos_three_modules() -> List[Dict[str, Any]]:
    return [
        {"id": "CO-01", "statement": "Alpha skill one.", "source_chunk_ids": ["a1"]},
        {"id": "CO-02", "statement": "Alpha skill two.", "source_chunk_ids": ["a2"]},
        {"id": "CO-03", "statement": "Bravo skill one.", "source_chunk_ids": ["b1"]},
        {"id": "CO-04", "statement": "Bravo skill two.", "source_chunk_ids": ["b2"]},
        {"id": "CO-05", "statement": "Charlie skill one.", "source_chunk_ids": ["c1"]},
        {"id": "CO-06", "statement": "Charlie skill two.", "source_chunk_ids": ["c2"]},
    ]


class _AnchorProvider:
    """Records ``author_terminal_for_cluster`` calls; optionally arms the stop."""

    def __init__(self, *, arm_after: int = 0) -> None:
        self._model = "test-model-v1"
        self._arm_after = arm_after
        self.author_calls: List[int] = []

    def author_terminal_for_cluster(
        self, cluster_cos, *, course_name, cluster_index
    ):
        self.author_calls.append(cluster_index)
        if self._arm_after and len(self.author_calls) == self._arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return {"statement": f"TO for module {cluster_index}.", "bloom_level": "apply"}


def _derive(provider, *, cluster_cp: Optional[Path] = None, embed=None):
    all_chunks = _three_module_all_chunks()
    chunks_by_id = {c["id"]: c for c in all_chunks}
    return pt._derive_terminals_chapter_anchored(
        provider=provider,
        chapter_cos=_cos_three_modules(),
        chunks_by_id=chunks_by_id,
        all_chunks=all_chunks,
        embed=embed,
        course_name="FXMATH_101",
        mint_lo_id=_mint,
        capture=None,
        checkpoint_path=cluster_cp,
    )


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# ---------------------------------------------------------------------------
# (1) chapter_anchored signal + book-order TOs + provenance stamps
# ---------------------------------------------------------------------------
def test_chapter_anchored_signal_and_book_order():
    provider = _AnchorProvider()
    terminals, signals = _derive(provider)

    assert signals["to_derivation"] == "chapter_anchored"
    assert signals["chapter_anchor_modules"] == 3
    assert len(terminals) == 3
    # TOs minted in book (module) order with the anchor provenance stamped.
    assert [t["id"] for t in terminals] == ["TO-01", "TO-02", "TO-03"]
    assert [t["anchor_module_id"] for t in terminals] == ["mod-a", "mod-b", "mod-c"]
    assert [t["anchor_module_title"] for t in terminals] == ["MOD-A", "MOD-B", "MOD-C"]
    # Every CO's terminal_id was set in-loop from its module.
    cos = _cos_three_modules()
    # Re-derive to inspect the mutated COs (fresh provider, fresh CO list).
    prov2 = _AnchorProvider()
    all_chunks = _three_module_all_chunks()
    pt._derive_terminals_chapter_anchored(
        provider=prov2,
        chapter_cos=cos,
        chunks_by_id={c["id"]: c for c in all_chunks},
        all_chunks=all_chunks,
        embed=None,
        course_name="FXMATH_101",
        mint_lo_id=_mint,
    )
    assert cos[0]["terminal_id"] == "TO-01"  # mod-a
    assert cos[2]["terminal_id"] == "TO-02"  # mod-b
    assert cos[4]["terminal_id"] == "TO-03"  # mod-c
    # to_source_chunk_assignments unions the members' cited chunks.
    assert signals["to_source_chunk_assignments"]["TO-01"] == ["a1", "a2"]


# ---------------------------------------------------------------------------
# (2) backlink is a NO-OP on the pre-set terminal_id (anchor honored)
# ---------------------------------------------------------------------------
def test_backlink_noop_on_preset_terminal_id():
    from lib.ontology.lo_backlink import backlink_cos_to_tos

    provider = _AnchorProvider()
    all_chunks = _three_module_all_chunks()
    cos = _cos_three_modules()
    terminals, _sig = pt._derive_terminals_chapter_anchored(
        provider=provider,
        chapter_cos=cos,
        chunks_by_id={c["id"]: c for c in all_chunks},
        all_chunks=all_chunks,
        embed=FakeEmbed(),
        course_name="FXMATH_101",
        mint_lo_id=_mint,
    )
    before = [c["terminal_id"] for c in cos]
    # reassign forced off (as the call site does in anchor mode) → pure no-op.
    backlink_cos_to_tos(terminals, cos, embed=FakeEmbed(), reassign=False)
    after = [c["terminal_id"] for c in cos]
    assert before == after  # the anchor assignment survived the backlink


# ---------------------------------------------------------------------------
# (3) graceful degrade — single module → ([], reason) so caller falls back
# ---------------------------------------------------------------------------
def test_degrade_single_module_returns_reason():
    provider = _AnchorProvider()
    all_chunks = [_chunk("a1", "mod-a"), _chunk("a2", "mod-a")]
    cos = [
        {"statement": "one", "source_chunk_ids": ["a1"]},
        {"statement": "two", "source_chunk_ids": ["a2"]},
    ]
    terminals, signals = pt._derive_terminals_chapter_anchored(
        provider=provider,
        chapter_cos=cos,
        chunks_by_id={c["id"]: c for c in all_chunks},
        all_chunks=all_chunks,
        embed=None,
        course_name="FXMATH_101",
        mint_lo_id=_mint,
    )
    assert terminals == []
    assert "too_few_modules" in signals["to_derivation_degraded_reason"]
    assert provider.author_calls == []  # never authored


# ---------------------------------------------------------------------------
# (4) per-module sidecar reuse — resume skips authored modules
# ---------------------------------------------------------------------------
def test_sidecar_reuse_skips_authored_modules(tmp_path, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_clusters_checkpoint.jsonl"
    first = _AnchorProvider()
    terminals1, sig1 = _derive(first, cluster_cp=cp)
    assert len(first.author_calls) == 3
    assert sig1["to_clusters_reused_from_checkpoint"] == 0

    # Re-run against the SAME sidecar → zero author calls, all reused.
    second = _AnchorProvider()
    terminals2, sig2 = _derive(second, cluster_cp=cp)
    assert second.author_calls == []
    assert sig2["to_clusters_reused_from_checkpoint"] == 3
    assert [t["statement"] for t in terminals2] == [
        t["statement"] for t in terminals1
    ]


# ---------------------------------------------------------------------------
# (5) graceful stop propagates at the module boundary (parity w/ bottom-up)
# ---------------------------------------------------------------------------
def test_stop_between_modules_exact_n(tmp_path, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_clusters_checkpoint.jsonl"
    provider = _AnchorProvider(arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _derive(provider, cluster_cp=cp)

    # Module 3's loop-top check_stop raised: exactly 2 authored + on disk.
    assert len(provider.author_calls) == 2
    store = pt._stage2_cluster_store(cp)
    assert len(store.load()) == 2


def test_stop_resume_completes(tmp_path, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_clusters_checkpoint.jsonl"
    interrupted = _AnchorProvider(arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _derive(interrupted, cluster_cp=cp)
    assert len(interrupted.author_calls) == 2

    stop_control.clear_stop(include_global=True)
    resume = _AnchorProvider()  # never arms
    terminals, _sig = _derive(resume, cluster_cp=cp)
    # Only the 1 un-authored module re-runs; 3 TOs emitted overall.
    assert len(resume.author_calls) == 1
    assert len(interrupted.author_calls) + len(resume.author_calls) == 3
    assert len(terminals) == 3


def test_stop_pre_armed_zero_author_calls(tmp_path, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_clusters_checkpoint.jsonl"
    stop_control.request_stop(scope="run", reason="test", source="test")
    provider = _AnchorProvider()
    with pytest.raises(GracefulStopRequested):
        _derive(provider, cluster_cp=cp)
    assert provider.author_calls == []


# ---------------------------------------------------------------------------
# (6) call-site selection wiring through _run_stage2_window_synthesis
# ---------------------------------------------------------------------------
class _WindowAnchorProvider(_AnchorProvider):
    """Window + cluster + reconcile surface; one candidate per window."""

    def __init__(self) -> None:
        super().__init__()
        self._max_tokens = 0  # shrink reserve → 1 chunk per window

    def batch_chapters(self, items, batch_size: int = 10):
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def synthesize_window_objectives(
        self, window, *, course_name, draft_terminal_objectives
    ):
        chunk_ids = list(window.get("chunk_ids") or [])
        cid = str(window.get("chapter_id"))
        return {
            "candidate_objectives": [
                {
                    "statement": f"Objective for {cid} citing {chunk_ids}.",
                    "source_chunk_ids": chunk_ids,
                }
            ]
        }

    def reconcile_terminal_objectives(self, draft_tos, chapter_cos, *, course_name):
        return {"terminal_objectives": [{"id": "TO-01", "statement": "Reconciled."}]}


def _window_chapters() -> List[Dict[str, Any]]:
    # 3 chapters 1:1 with the 3 modules; order_fallback proportions by length.
    return [
        {"id": "mod-a", "chapter_text": "x" * 10},
        {"id": "mod-b", "chapter_text": "y" * 10},
        {"id": "mod-c", "chapter_text": "z" * 10},
    ]


def _run_window(*, anchor_on: bool, monkeypatch):
    import lib.embedding.providers as _providers
    import lib.objectives.objective_grounding as _og

    class _Passthrough:
        def __init__(self, grounded):
            self.grounded = list(grounded)
            self.ungrounded: List[Dict[str, Any]] = []
            self.available = False
            self.dropped_count = 0
            self.reground_count = 0

    monkeypatch.setattr(
        _providers, "build_embedding_client", lambda *a, **k: FakeEmbed()
    )
    monkeypatch.setattr(
        _og,
        "ground_candidates",
        lambda cands, chunks_by_id, require=False: _Passthrough(cands),
    )
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", "80")
    if anchor_on:
        monkeypatch.setenv("ED4ALL_TO_CHAPTER_ANCHOR", "1")
    else:
        monkeypatch.delenv("ED4ALL_TO_CHAPTER_ANCHOR", raising=False)

    all_chunks = _three_module_all_chunks()
    chunks_by_id = {c["id"]: c for c in all_chunks}
    provider = _WindowAnchorProvider()
    return provider, asyncio.run(
        pt._run_stage2_window_synthesis(
            provider=provider,
            provider_error=_ProviderError,
            chapters=_window_chapters(),
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
        )
    )


def test_call_site_flag_on_selects_chapter_anchored(monkeypatch):
    _prov, result = _run_window(anchor_on=True, monkeypatch=monkeypatch)
    _cos, terminals, _mint_method, signals = result
    assert signals["to_derivation"] == "chapter_anchored"
    assert signals["chapter_anchor_modules"] >= 2
    assert all("anchor_module_id" in t for t in terminals)


def test_call_site_flag_off_selects_bottom_up_untouched(monkeypatch):
    prov, result = _run_window(anchor_on=False, monkeypatch=monkeypatch)
    _cos, terminals, _mint_method, signals = result
    # Flag OFF → bottom-up path (byte-identical to today); NO anchor signal.
    assert signals["to_derivation"] == "bottom_up"
    assert "chapter_anchor_modules" not in signals
    assert all("anchor_module_id" not in t for t in terminals)
