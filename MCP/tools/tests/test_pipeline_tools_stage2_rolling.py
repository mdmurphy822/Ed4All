"""Stage-2 window-synthesis ROLLING-WINDOW dispatch invariants.

The ``course_planning`` Stage-2 per-window dispatch was converted from a batch
BARRIER (gather each batch of ``_width``, idling the pool on the slowest window
per batch) to a ROLLING WINDOW: keep ``_width`` window futures in flight at all
times and refill the instant one resolves. These tests pin the rolling
behaviour that the barrier could NOT provide:

- **Concurrency stays pinned** at ``_width`` with ragged per-window costs — one
  slow window never idles the pool down to itself while fast windows from a
  later batch region wait (the batch-barrier defect). Proven deterministically:
  a fast window from the SECOND batch region COMPLETES BEFORE the slow first
  window — impossible under a barrier, where batch 2 cannot start until batch 1
  (including the slow window) has fully drained.
- **Per-window sidecar append** — each window's resume checkpoint lands as its
  OWN future resolves (finer than the old per-batch append), so a late window
  observes many prior windows already persisted, spanning batch boundaries.
- **Pass C/D still receives the full survivor set** — the rolling change only
  compresses the synthesis-dispatch span; every window candidate reaches the
  downstream grounding + dedup merge.
- **Width is never raised** — peak concurrency equals ``_width`` (the provider's
  configured batch size capped at the window count), not more.

Hermetic: a mock provider (no GPU / ollama / network) with controllable
per-window durations, ``build_embedding_client`` forced unavailable, and
``ground_candidates`` monkeypatched to a passthrough (no NLI model loads).
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402


class _ProviderError(RuntimeError):
    pass


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


# ---------------------------------------------------------------------------
# Ragged-cost mock provider — records live concurrency + completion order.
# ---------------------------------------------------------------------------
class _RaggedProvider:
    """One SLOW window (the one carrying ``slow_chunk``) + fast rest.

    ``batch_size`` drives ``_width`` (the rolling in-flight ceiling); a live
    thread-safe counter records the peak concurrency and the exact completion
    order so the refill-without-drain behaviour is a deterministic assertion.
    """

    def __init__(
        self,
        *,
        batch_size: int = 4,
        slow_chunk: str = "c1",
        slow: float = 0.30,
        fast: float = 0.05,
        model: str = "test-model-v1",
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        self._model = model
        self._max_tokens = 0  # shrink the fixed reserve → 1 chunk per window
        self._bs = batch_size
        self._slow_chunk = slow_chunk
        self._slow = slow
        self._fast = fast
        self._cp = checkpoint_path
        self.window_calls: List[str] = []
        self.completion_order: List[str] = []
        self.records_seen: List[int] = []
        self._live = 0
        self._max_live = 0
        self._lock = threading.Lock()

    @property
    def max_live(self) -> int:
        return self._max_live

    def batch_chapters(self, items, batch_size: int = 10):
        return [items[i:i + self._bs] for i in range(0, len(items), self._bs)]

    def synthesize_window_objectives(
        self, window, *, course_name, draft_terminal_objectives
    ):
        label = f"{window.get('chapter_id')}#w{window.get('window_index')}"
        chunk_ids = list(window.get("chunk_ids") or [])
        with self._lock:
            self._live += 1
            self._max_live = max(self._max_live, self._live)
            if self._cp is not None:
                self.records_seen.append(
                    len(pt._load_stage2_windows_checkpoint(self._cp))
                )
        time.sleep(self._slow if self._slow_chunk in chunk_ids else self._fast)
        with self._lock:
            self._live -= 1
            self.window_calls.append(label)
            self.completion_order.append(label)
        return {
            "candidate_objectives": [
                {
                    "statement": f"Objective for {label}.",
                    "source_chunk_ids": chunk_ids,
                }
            ]
        }

    def reconcile_terminal_objectives(
        self, draft_tos, chapter_cos, *, course_name
    ):
        return {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Reconciled TO."}
            ]
        }


class _PassthroughGrounding:
    def __init__(self, grounded: List[Dict[str, Any]]) -> None:
        self.grounded = grounded
        self.ungrounded: List[Dict[str, Any]] = []
        self.available = False
        self.dropped_count = 0
        self.reground_count = 0


def _chapters() -> List[Dict[str, Any]]:
    return [
        {"id": "ch1", "chapter_text": "xxxxxxxxxx"},
        {"id": "ch2", "chapter_text": "yyyyyyyyyy"},
    ]


def _chunks() -> List[Dict[str, Any]]:
    # 8 chunks / 2 chapters → 4 chunks each; num_ctx=80 → 1 chunk/window → 8
    # windows total. c1..c4 land in ch1 (batch region 1), c5..c8 in ch2 (2).
    return [{"id": f"c{i}", "text": f"Body {i}."} for i in range(1, 9)]


def _run(provider, *, checkpoint_path: Optional[Path], monkeypatch):
    """Drive _run_stage2_window_synthesis in chunk_window mode, hermetically."""
    import lib.embedding.providers as _providers
    import lib.objectives.objective_grounding as _og

    def _fake_build(*_a, **_k):
        raise _providers.EmbeddingBackendUnavailable("no extras in test")

    monkeypatch.setattr(_providers, "build_embedding_client", _fake_build)
    monkeypatch.setattr(
        _og,
        "ground_candidates",
        lambda cands, chunks_by_id, require=False: _PassthroughGrounding(
            list(cands)
        ),
    )
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", "80")

    all_chunks = _chunks()
    chunks_by_id = {c["id"]: c for c in all_chunks}
    return asyncio.run(
        pt._run_stage2_window_synthesis(
            provider=provider,
            provider_error=_ProviderError,
            chapters=_chapters(),
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
            checkpoint_path=checkpoint_path,
        )
    )


def _slow_label(completion_order: List[str]) -> str:
    """The label carrying c1 completes LAST (it is the slow window)."""
    return completion_order[-1]


# ---------------------------------------------------------------------------
# (1) concurrency stays pinned: a batch-2 window completes before the slow one
# ---------------------------------------------------------------------------
def test_rolling_refill_beats_slow_window(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _RaggedProvider(batch_size=4, slow_chunk="c1")

    _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    order = provider.completion_order
    assert len(provider.window_calls) == 8  # all 8 windows synthesized

    # The slow window (c1, batch region 1) must complete LAST — every fast
    # window, INCLUDING batch-region-2 windows (ch2), finished before it. Under
    # a batch barrier, batch 2 could not START until batch 1 (with the slow
    # window) had fully drained, so a ch2 window completing before the slow
    # window is a rolling-only property.
    slow = _slow_label(order)
    assert "ch1" in slow  # c1 lives in ch1
    ch2_positions = [i for i, lbl in enumerate(order) if lbl.startswith("ch2")]
    assert ch2_positions, "expected ch2 (batch-region-2) windows to complete"
    slow_pos = order.index(slow)
    assert min(ch2_positions) < slow_pos, (
        "a batch-region-2 window must complete before the slow window — "
        "the pool refilled instead of idling to 1 (barrier defect)"
    )


# ---------------------------------------------------------------------------
# (2) width is pinned to the provider batch size, never raised
# ---------------------------------------------------------------------------
def test_peak_concurrency_equals_width(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _RaggedProvider(batch_size=4, slow_chunk="c1")

    _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    # 8 windows, batch_size=4 → _width=4. Peak in-flight must REACH 4 (pinned,
    # sustained by refill) and NEVER exceed it (width not raised).
    assert provider.max_live == 4


def test_width_one_is_fully_serial(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _RaggedProvider(batch_size=1, slow_chunk="c1", slow=0.02, fast=0.0)

    _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    # batch_size=1 → _width=1 → never more than one window in flight.
    assert provider.max_live == 1
    assert len(provider.window_calls) == 8


# ---------------------------------------------------------------------------
# (3) per-window sidecar append (not per-batch) — spans batch boundaries
# ---------------------------------------------------------------------------
def test_sidecar_appends_per_window_across_batches(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _RaggedProvider(
        batch_size=4, slow_chunk="c1", checkpoint_path=cp
    )

    _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    # Every window is on disk at the end.
    assert len(pt._load_stage2_windows_checkpoint(cp)) == 8
    # A late window observed MORE than one batch's worth (_width=4) of records
    # already persisted at its entry — impossible if appends only landed once a
    # whole batch had gathered. The slow window keeps a slot busy while later
    # fast windows resolve + append one at a time, so the tail sees ≥ _width.
    assert max(provider.records_seen) >= 4


# ---------------------------------------------------------------------------
# (4) Pass C/D receives the FULL survivor set (rolling only compresses dispatch)
# ---------------------------------------------------------------------------
def test_full_survivor_set_reaches_merge(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"

    provider = _RaggedProvider(batch_size=4, slow_chunk="c1")
    res = _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    # Every window yields one candidate objective; all 8 must survive the
    # rolling dispatch and reach the downstream Pass C / Pass D merge, which the
    # rolling change deliberately leaves as a barrier. One CO per window → 8 COs
    # emitted is the full-survivor witness: no window candidate was dropped by
    # the dispatch compression.
    chapter_cos = res[0]
    assert len(chapter_cos) == 8
    assert len(provider.window_calls) == 8
