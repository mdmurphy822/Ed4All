"""Stage-2 per-window resume-checkpoint sidecar invariants.

A killed / timed-out ``course_planning`` phase must not lose the expensive
per-window ``synthesize_window_objectives`` 7B calls on resume. These tests
exercise the sidecar mechanism wired into
``MCP.tools.pipeline_tools._run_stage2_window_synthesis`` (chunk_window path)
plus its module-level helpers.

Hermetic: a mock provider (no GPU / ollama / network), ``build_embedding_client``
forced unavailable (legacy reconcile path, no embeddings), and
``ground_candidates`` monkeypatched to a passthrough so no NLI model loads and
the downstream is a deterministic function of the candidate set.
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


class _ProviderError(RuntimeError):
    pass


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _WindowProvider:
    """Mock stage-2 provider for the chunk_window dispatch path.

    ``synthesize_window_objectives`` records every call and returns one candidate
    objective per window (deterministic, keyed on the window label).
    """

    def __init__(self, *, model: str = "test-model-v1") -> None:
        self._model = model
        self._max_tokens = 0  # shrink the fixed reserve → 1 chunk per window
        self.window_calls: List[str] = []

    def batch_chapters(self, items, batch_size: int = 10):
        return [
            items[i:i + batch_size] for i in range(0, len(items), batch_size)
        ]

    def synthesize_window_objectives(
        self, window, *, course_name, draft_terminal_objectives
    ):
        cid = str(window.get("chapter_id"))
        widx = window.get("window_index")
        self.window_calls.append(f"{cid}#w{widx}")
        return {
            "candidate_objectives": [
                {
                    "statement": f"Objective for {cid} window {widx}.",
                    # Non-resolving id keeps this hermetic AND the passthrough
                    # ground_candidates keeps it in the pool.
                    "source_chunk_ids": list(window.get("chunk_ids") or []),
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
    # No source_file → order_fallback partition; equal chapter_text → 2 chunks
    # each. A small num_ctx makes every chunk its own window (4 windows total).
    return [
        {"id": "ch1", "chapter_text": "xxxxxxxxxx"},
        {"id": "ch2", "chapter_text": "yyyyyyyyyy"},
    ]


def _chunks(mutate_text: bool = False) -> List[Dict[str, Any]]:
    suffix = " EDITED" if mutate_text else ""
    return [
        {"id": "c1", "text": "Alpha body." + suffix},
        {"id": "c2", "text": "Bravo body." + suffix},
        {"id": "c3", "text": "Charlie body." + suffix},
        {"id": "c4", "text": "Delta body." + suffix},
    ]


def _run(
    provider,
    *,
    checkpoint_path: Optional[Path],
    monkeypatch,
    mutate_text: bool = False,
):
    """Drive _run_stage2_window_synthesis in chunk_window mode, hermetically."""
    import lib.embedding.providers as _providers
    import lib.objectives.objective_grounding as _og

    def _fake_build(*_a, **_k):
        raise _providers.EmbeddingBackendUnavailable("no extras in test")

    monkeypatch.setattr(_providers, "build_embedding_client", _fake_build)
    # Passthrough grounding: keep all candidates, load no NLI model.
    monkeypatch.setattr(
        _og,
        "ground_candidates",
        lambda cands, chunks_by_id, require=False: _PassthroughGrounding(
            list(cands)
        ),
    )
    # Force one chunk per window so a 4-chunk corpus → 4 windows.
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", "80")

    all_chunks = _chunks(mutate_text=mutate_text)
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
            course_name="MATH_101",
            provider_env="local",
            chapter_synthesis_failures=[],
            mint_lo_id=_mint,
            kwargs={},
            capture=None,
            checkpoint_path=checkpoint_path,
        )
    )


# ---------------------------------------------------------------------------
# (a) sidecar written per window
# ---------------------------------------------------------------------------
def test_sidecar_written_per_window(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _WindowProvider()
    _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    assert len(provider.window_calls) == 4  # 4 windows dispatched
    assert cp.exists()
    records = pt._load_stage2_windows_checkpoint(cp)
    assert len(records) == 4
    for rec in records.values():
        assert rec["schema_version"] == pt._STAGE2_CHECKPOINT_SCHEMA_VERSION
        assert isinstance(rec["fingerprint"], str) and rec["fingerprint"]
        cands = rec["candidate_objectives"]
        assert cands and isinstance(cands, list)
        assert cands[0]["statement"].startswith("Objective for ")


# ---------------------------------------------------------------------------
# (b) resume skips matching windows — provider call-count drops, byte-equivalent
# ---------------------------------------------------------------------------
def test_resume_skips_matching_windows_byte_equivalent(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"

    provider1 = _WindowProvider()
    res_fresh = _run(provider1, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider1.window_calls) == 4

    # Resume against the populated sidecar: every window fingerprint matches, so
    # ZERO provider dispatches — and the full return is byte-equivalent.
    provider2 = _WindowProvider()
    res_resume = _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert provider2.window_calls == []  # nothing re-dispatched

    # The synthesized output (chapter_cos, terminals, mint_method) is
    # byte-equivalent to the uninterrupted run — the candidate list that
    # matters is identical.
    assert res_resume[:3] == res_fresh[:3]
    # The grounding_signals differ ONLY on the reuse counters (correctly): the
    # fresh run reused 0 windows, the resume reused all 4.
    fresh_sig, resume_sig = dict(res_fresh[3]), dict(res_resume[3])
    for k in ("windows_reused_from_checkpoint",):
        fresh_sig.pop(k, None)
        resume_sig.pop(k, None)
    assert fresh_sig == resume_sig
    assert res_fresh[3]["windows_reused_from_checkpoint"] == 0
    assert res_resume[3]["windows_total"] == 4
    assert res_resume[3]["windows_reused_from_checkpoint"] == 4


def test_partial_resume_dispatches_only_missing(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider1 = _WindowProvider()
    _run(provider1, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider1.window_calls) == 4

    # Simulate a partial run: drop one window record from the sidecar.
    lines = cp.read_text(encoding="utf-8").splitlines()
    cp.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    provider2 = _WindowProvider()
    res = _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider2.window_calls) == 1  # only the dropped window re-ran
    assert res[3]["windows_reused_from_checkpoint"] == 3


# ---------------------------------------------------------------------------
# (c) fingerprint mismatch re-runs
# ---------------------------------------------------------------------------
def test_fingerprint_mismatch_model_reruns(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_WindowProvider(model="test-model-v1"),
         checkpoint_path=cp, monkeypatch=monkeypatch)

    # A different model id changes every fingerprint → full re-dispatch.
    provider2 = _WindowProvider(model="test-model-v2")
    res = _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider2.window_calls) == 4
    assert res[3]["windows_reused_from_checkpoint"] == 0


def test_fingerprint_mismatch_changed_chunk_text_reruns(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_WindowProvider(), checkpoint_path=cp, monkeypatch=monkeypatch)

    # Same windows/keys, but each chunk's body changed → fingerprints mismatch.
    provider2 = _WindowProvider()
    res = _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch,
               mutate_text=True)
    assert len(provider2.window_calls) == 4
    assert res[3]["windows_reused_from_checkpoint"] == 0


# ---------------------------------------------------------------------------
# (d) torn trailing line tolerated
# ---------------------------------------------------------------------------
def test_torn_trailing_line_tolerated(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_WindowProvider(), checkpoint_path=cp, monkeypatch=monkeypatch)

    # Simulate a crash mid-append: a half-written final JSON line.
    with cp.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "chapter_id": "ch2", "window_ind')

    # Loader skips the torn line without raising; the 4 valid records survive.
    records = pt._load_stage2_windows_checkpoint(cp)
    assert len(records) == 4

    # A resume still fully reuses (torn line ignored, no re-dispatch).
    provider2 = _WindowProvider()
    res = _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert provider2.window_calls == []
    assert res[3]["windows_reused_from_checkpoint"] == 4


# ---------------------------------------------------------------------------
# (e) flag opt-out disables both write and reuse
# ---------------------------------------------------------------------------
def test_opt_out_disables_write(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    monkeypatch.setenv("ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT", "0")
    provider = _WindowProvider()
    _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider.window_calls) == 4
    assert not cp.exists()  # no sidecar written when disabled


def test_opt_out_disables_reuse(tmp_path, monkeypatch):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    # First: a normal enabled run populates the sidecar.
    _run(_WindowProvider(), checkpoint_path=cp, monkeypatch=monkeypatch)
    assert cp.exists()

    # Then: with the flag off, a resume ignores the sidecar and re-dispatches.
    monkeypatch.setenv("ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT", "0")
    provider2 = _WindowProvider()
    res = _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider2.window_calls) == 4
    assert res[3]["windows_reused_from_checkpoint"] == 0


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------
def test_fingerprint_is_deterministic_and_sensitive():
    base = dict(
        chunk_ids=["c1", "c2"],
        chunks=[{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}],
        draft_block="draft",
        model="m1",
        num_ctx=4096,
        system_prompt="sys",
    )
    fp = pt._stage2_window_fingerprint(**base)
    assert fp == pt._stage2_window_fingerprint(**base)  # deterministic
    # chunk_ids order does not matter (sorted), but membership does.
    reordered = dict(base, chunk_ids=["c2", "c1"])
    assert pt._stage2_window_fingerprint(**reordered) == fp
    # Each input axis flips the fingerprint.
    for key, val in [
        ("chunk_ids", ["c1", "c3"]),
        ("chunks", [{"id": "c1", "text": "a"}, {"id": "c2", "text": "B!"}]),
        ("draft_block", "draft2"),
        ("model", "m2"),
        ("num_ctx", 8192),
        ("system_prompt", "sys2"),
    ]:
        assert pt._stage2_window_fingerprint(**dict(base, **{key: val})) != fp


def test_checkpoint_enabled_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT", raising=False)
    assert pt._objective_synthesis_checkpoint_enabled() is True  # default ON
    for tok in ("garbage", "1", "true", "yes", "on"):
        monkeypatch.setenv("ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT", tok)
        assert pt._objective_synthesis_checkpoint_enabled() is True
    for tok in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT", tok)
        assert pt._objective_synthesis_checkpoint_enabled() is False


def test_loader_missing_file_returns_empty(tmp_path):
    assert pt._load_stage2_windows_checkpoint(None) == {}
    assert pt._load_stage2_windows_checkpoint(tmp_path / "nope.jsonl") == {}
