"""Stage-2 chapter_fallback graceful-stop + resume-checkpoint invariants (P4).

The ``course_planning`` Stage-2 chapter_fallback path (chunkset MISSING) runs a
legacy per-CHAPTER dispatch (``synthesize_chapter_objectives``). This wave gives
it a fingerprinted per-chapter resume sidecar
(``.stage2_chapter_fallback_checkpoint.jsonl``) + the P3.0 graceful-stop marker
pattern, mirroring the window sidecar a few branches above.

Two axes are exercised against the SAME sidecar:

* the FULL canonical checkpoint invariant set — per-unit write, resume
  byte-equivalence, fingerprint-mismatch re-run, torn trailing line tolerated,
  family-flag opt-out;
* the three graceful-stop legs — stop after N (sidecar == provider == N),
  resume (total-N calls, byte-equivalent), pre-armed sentinel (0 calls) — plus
  the P3.0 in-flight marker sweep (``sidecar records == provider calls``).

Hermetic: a mock chapter provider (no GPU / ollama / network),
``build_embedding_client`` forced unavailable (→ the reconcile TO fallback),
``ground_candidates`` monkeypatched to a passthrough (no NLI model). Sentinel
isolation via ``state_runs_isolated`` (per-test ``ED4ALL_STATE_RUNS_DIR``) + a
synthetic ``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402

_RUN_ID = "STOP_CHFB_TESTRUN"
_CKPT_NAME = pt._STAGE2_CHAPTER_FALLBACK_CHECKPOINT_NAME


class _ProviderError(RuntimeError):
    pass


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _ChapterProvider:
    """Records every per-chapter call; one candidate objective per chapter."""

    def __init__(self, *, model: str = "test-model-v1") -> None:
        self._model = model
        self.chapter_calls: List[str] = []

    def batch_chapters(self, items, batch_size: int = 10):
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def synthesize_chapter_objectives(
        self, chapter, *, course_name, draft_terminal_objectives
    ):
        cid = str(chapter.get("id"))
        self.chapter_calls.append(cid)
        return {
            "chapter_id": cid,
            "chapter_objectives": [
                {
                    "statement": f"Objective for chapter {cid}.",
                    "bloom_level": "understand",
                }
            ],
        }

    def reconcile_terminal_objectives(self, draft_tos, chapter_cos, *, course_name):
        return {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Reconciled TO."}
            ]
        }


class _ArmAfterChapterProvider(_ChapterProvider):
    """Arms the REAL run-scoped sentinel AFTER its ``arm_after``-th call.

    With ``batch_size=1`` this drives the BETWEEN-batches path: chapter N
    completes + is appended, then the next batch-loop-top ``check_stop`` raises.
    """

    def __init__(self, *, arm_after: int, batch_size: int = 1) -> None:
        super().__init__()
        self._arm_after = arm_after
        self._batch_size = batch_size

    def batch_chapters(self, items, batch_size: int = 10):
        return super().batch_chapters(items, batch_size=self._batch_size)

    def synthesize_chapter_objectives(self, chapter, **kw):
        out = super().synthesize_chapter_objectives(chapter, **kw)
        if len(self.chapter_calls) == self._arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return out


class _GatedStopProbe:
    """Deterministic in-flight ``stop_requested`` gate (monkeypatched onto pt).

    The first ``false_for`` probes return ``False`` (those chapters dispatch);
    every later probe returns ``True`` (those get ``STOP_MARKER``). Thread-safe
    so the completed/marker SPLIT is exact regardless of executor interleaving.
    """

    def __init__(self, false_for: int) -> None:
        self.false_for = false_for
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, run_id: Optional[str] = None) -> bool:
        with self._lock:
            self.calls += 1
            return self.calls > self.false_for


class _PassthroughGrounding:
    def __init__(self, grounded: List[Dict[str, Any]]) -> None:
        self.grounded = grounded
        self.ungrounded: List[Dict[str, Any]] = []
        self.available = False
        self.dropped_count = 0
        self.reground_count = 0


def _chapters(n: int = 4) -> List[Dict[str, Any]]:
    # Distinct chapter_text per chapter → distinct fingerprints.
    return [
        {"id": f"ch{i}", "chapter_text": f"Chapter {i} body " + chr(ord("a") + i) * 8}
        for i in range(1, n + 1)
    ]


def _run(provider, *, checkpoint_path: Path, monkeypatch, chapters=None):
    """Drive _run_stage2_window_synthesis in chapter_fallback mode, hermetically.

    ``checkpoint_path`` is the WINDOW sidecar path; the chapter_fallback sidecar
    is derived as its sibling (``checkpoint_path.parent / _CKPT_NAME``), matching
    production.
    """
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

    chapters = chapters if chapters is not None else _chapters()
    return asyncio.run(
        pt._run_stage2_window_synthesis(
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
            checkpoint_path=checkpoint_path,
        )
    )


def _cf_sidecar(cp: Path) -> Path:
    return cp.parent / _CKPT_NAME


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    """Per-test sentinel isolation: tmp state/runs + a synthetic run_id."""
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# ===========================================================================
# Canonical checkpoint invariants
# ===========================================================================
def test_sidecar_written_per_chapter(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _ChapterProvider()
    _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    assert len(provider.chapter_calls) == 4
    records = pt._load_stage2_chapter_fallback_checkpoint(_cf_sidecar(cp))
    assert set(records) == {"ch1", "ch2", "ch3", "ch4"}
    for rec in records.values():
        assert isinstance(rec["fingerprint"], str) and rec["fingerprint"]
        assert isinstance(rec["chapter_objectives"], list)


def test_resume_skips_matching_chapters_byte_equivalent(
    tmp_path, monkeypatch, _armed_env
):
    # Uninterrupted baseline (its OWN sidecar) for the byte-equivalence oracle.
    base_cp = tmp_path / "base" / ".stage2_windows_checkpoint.jsonl"
    res_full = _run(_ChapterProvider(), checkpoint_path=base_cp,
                    monkeypatch=monkeypatch)

    # Populate a fresh sidecar, then resume against it: every chapter
    # fingerprint matches → ZERO provider dispatches, byte-equivalent output.
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_ChapterProvider(), checkpoint_path=cp, monkeypatch=monkeypatch)
    resume = _ChapterProvider()
    res_resume = _run(resume, checkpoint_path=cp, monkeypatch=monkeypatch)

    assert resume.chapter_calls == []  # nothing re-dispatched
    assert res_resume[:3] == res_full[:3]  # canonical COs / TOs / mint method


def test_partial_resume_dispatches_only_missing(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    # First run authors ch1+ch2 only, then a resume authors ch3+ch4.
    _run(_ChapterProvider(), checkpoint_path=cp, monkeypatch=monkeypatch,
         chapters=_chapters(4)[:2])
    provider2 = _ChapterProvider()
    _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch,
         chapters=_chapters(4))
    # ch1/ch2 reused from the sidecar; only ch3/ch4 dispatched.
    assert sorted(provider2.chapter_calls) == ["ch3", "ch4"]
    records = pt._load_stage2_chapter_fallback_checkpoint(_cf_sidecar(cp))
    assert set(records) == {"ch1", "ch2", "ch3", "ch4"}


def test_fingerprint_mismatch_model_reruns(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_ChapterProvider(model="test-model-v1"),
         checkpoint_path=cp, monkeypatch=monkeypatch)
    # A different model id changes every fingerprint → full re-dispatch.
    provider2 = _ChapterProvider(model="test-model-v2")
    _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider2.chapter_calls) == 4


def test_fingerprint_mismatch_changed_chapter_text_reruns(
    tmp_path, monkeypatch, _armed_env
):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_ChapterProvider(), checkpoint_path=cp, monkeypatch=monkeypatch)
    # Same chapter ids, but each chapter body changed → fingerprints mismatch.
    mutated = [dict(c, chapter_text=c["chapter_text"] + " EDIT") for c in _chapters()]
    provider2 = _ChapterProvider()
    _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch, chapters=mutated)
    assert len(provider2.chapter_calls) == 4


def test_torn_trailing_line_tolerated(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_ChapterProvider(), checkpoint_path=cp, monkeypatch=monkeypatch)
    sidecar = _cf_sidecar(cp)
    # Append a half-written (torn) trailing line, as a crash mid-append would.
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "ch5", "fingerpr')
    records = pt._load_stage2_chapter_fallback_checkpoint(sidecar)
    assert set(records) == {"ch1", "ch2", "ch3", "ch4"}  # torn line skipped
    # A resume still fully reuses (torn line ignored, no re-dispatch).
    provider2 = _ChapterProvider()
    _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert provider2.chapter_calls == []


def test_family_flag_opt_out_disables(tmp_path, monkeypatch, _armed_env):
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    _run(_ChapterProvider(), checkpoint_path=cp, monkeypatch=monkeypatch)
    # No sidecar written when the family flag is off.
    assert not _cf_sidecar(cp).exists()
    # And a "resume" re-dispatches everything (no reuse).
    provider2 = _ChapterProvider()
    _run(provider2, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(provider2.chapter_calls) == 4


def test_fingerprint_is_deterministic_and_sensitive():
    base = dict(
        chapter_text="Some bounded chapter prose.",
        course_name="MATH_101",
        draft_block="  - Draft TO.",
        model="m1",
        num_ctx=4096,
        system_prompt="SYS",
    )
    fp = pt._stage2_chapter_fallback_fingerprint(**base)
    assert fp == pt._stage2_chapter_fallback_fingerprint(**base)  # deterministic
    for key, val in [
        ("chapter_text", "Different prose."),
        ("course_name", "PHYS_101"),
        ("draft_block", "  - Other."),
        ("model", "m2"),
        ("num_ctx", 8192),
        ("system_prompt", "SYS2"),
    ]:
        assert pt._stage2_chapter_fallback_fingerprint(
            **dict(base, **{key: val})
        ) != fp


def test_loader_missing_file_returns_empty(tmp_path):
    assert pt._load_stage2_chapter_fallback_checkpoint(None) == {}
    assert pt._load_stage2_chapter_fallback_checkpoint(tmp_path / "nope.jsonl") == {}


# ===========================================================================
# Graceful-stop legs
# ===========================================================================
def test_stop_between_batches_exact_n(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _ArmAfterChapterProvider(arm_after=2, batch_size=1)

    with pytest.raises(GracefulStopRequested):
        _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    # Chapter 3's loop-top check_stop raised: exactly 2 authored + on disk.
    assert len(provider.chapter_calls) == 2
    records = pt._load_stage2_chapter_fallback_checkpoint(_cf_sidecar(cp))
    assert len(records) == 2


def test_resume_after_stop_byte_equivalent(tmp_path, monkeypatch, _armed_env):
    # Uninterrupted baseline (own sidecar) for the byte-equivalence oracle.
    base_cp = tmp_path / "base" / ".stage2_windows_checkpoint.jsonl"
    res_full = _run(_ChapterProvider(), checkpoint_path=base_cp,
                    monkeypatch=monkeypatch)

    # Interrupted leg: stop after 2 of 4 chapters.
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    interrupted = _ArmAfterChapterProvider(arm_after=2, batch_size=1)
    with pytest.raises(GracefulStopRequested):
        _run(interrupted, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(interrupted.chapter_calls) == 2
    assert len(pt._load_stage2_chapter_fallback_checkpoint(_cf_sidecar(cp))) == 2

    # Resume leg: clear the sentinel, rerun the SAME sidecar → only the 2
    # un-attempted chapters dispatch; total across both legs == 4.
    stop_control.clear_stop(include_global=True)
    resume = _ChapterProvider()
    res_resume = _run(resume, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(resume.chapter_calls) == 2
    assert len(interrupted.chapter_calls) + len(resume.chapter_calls) == 4
    # Byte-equivalent to the uninterrupted run (canonical COs / TOs / mint).
    assert res_resume[:3] == res_full[:3]


def test_pre_armed_sentinel_zero_calls(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    stop_control.request_stop(scope="run", reason="test", source="test")

    provider = _ChapterProvider()
    with pytest.raises(GracefulStopRequested):
        _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    assert provider.chapter_calls == []  # nothing dispatched
    assert not _cf_sidecar(cp).exists()  # no sidecar written


def test_marker_pattern_in_flight_completed_persisted(
    tmp_path, monkeypatch, _armed_env
):
    """A stop armed mid-batch must not lose the batch's COMPLETED chapters.

    All 4 chapters dispatch in ONE batch; the deterministic gate lets the first
    2 through (they complete + append) and refuses the rest with STOP_MARKER. If
    ``_one_chapter`` RAISED instead of returning the marker, gather would
    propagate before the append loop and lose the 2 completed chapters. The
    invariant ``sidecar records == provider calls`` catches that regression.
    """
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    gate = _GatedStopProbe(false_for=2)
    # _one_chapter probes pt.stop_requested for the in-flight refusal; the batch
    # loop-top check_stop uses the REAL (un-armed) sentinel, so the raise comes
    # from the post-gather marker sweep.
    monkeypatch.setattr(pt, "stop_requested", gate)

    provider = _ChapterProvider()  # batch_size default 10 → all 4 in one batch
    with pytest.raises(GracefulStopRequested):
        _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    assert len(provider.chapter_calls) == 2
    records = pt._load_stage2_chapter_fallback_checkpoint(_cf_sidecar(cp))
    assert len(records) == len(provider.chapter_calls) == 2
