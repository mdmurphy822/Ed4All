"""Tests for the ``post_rewrite_validation`` prose-entailment overhaul on
``BlockProseEntailmentValidator`` — the three flagged features + singleton-hold.

Historically the gate scored every block's premise/hypothesis pairs SERIALLY
(one DeBERTa forward pass at a time) with no resume sidecar and no cooperative
stop, so a resume-loop cold-loaded the model repeatedly and never converged.
This overhaul adds, all behind flags (default-off = byte-identical serial):

1. **Cross-block process-pool scoring** (`ED4ALL_NLI_MICROBATCH_VALIDATORS`,
   repurposed; procs = `ED4ALL_NLI_VALIDATORS_PROCS`, default 4) — shard the
   scorable blocks across a spawn ``ProcessPoolExecutor`` whose workers each
   load their OWN NLI once via a picklable module-level factory.
2. **Per-block resume sidecar** (`ED4ALL_VALIDATION_CHECKPOINT`, default on) —
   content-addressed ``GroundednessReport`` cache next to the rewrite export.
3. **Stop-sentinel polling** — cooperative `ed4all stop` between blocks / before
   each shard submit, raising ``GracefulStopRequested`` after persisting.

CORRECTNESS IS PARAMOUNT (load-bearing quality gate): a deterministic per-pair
NLI stub (no DeBERTa load / GPU) is used and the tests assert **verdict identity**
(codes, severities, order, passed/action/score, decision order) between the
parallel path and the serial reference. The process pool is exercised via a
seam-injected inline executor (deterministic, exercising the REAL
``_pool_worker_init`` / ``_pool_score_task`` / factory-import code in-process),
a pickling-roundtrip test (the real cross-process risk), and a real spawn-pool
end-to-end smoke test (verdict identity; falls back to serial on any spawn
failure so it never flakes).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.retrieval.groundedness import score_groundedness  # noqa: E402
from lib.validators import block_prose_entailment as m  # noqa: E402
from lib.validators.block_prose_entailment import (  # noqa: E402
    _CODE_BLOCK_PROSE_CONTRADICTED,
    _CODE_BLOCK_PROSE_UNGROUNDED,
    _ENV_MICROBATCH_VALIDATORS,
    _ENV_NLI_VALIDATORS_FACTORY,
    _ENV_VALIDATION_CHECKPOINT,
    _ENV_VALIDATORS_PROCS,
    BlockProseEntailmentValidator,
    _microbatch_validators_enabled,
    _resolve_validators_procs,
)
from lib.validators.tests._prose_pool_stub import (  # noqa: E402
    MarkerStubNli,
    marker_stub_factory,
)

_STUB_FACTORY_DOTTED = "lib.validators.tests._prose_pool_stub:marker_stub_factory"


# --------------------------------------------------------------------- #
# Deterministic stubs
# --------------------------------------------------------------------- #


class _CountingNli:
    """Marker-keyed stub that counts ``score_batch`` calls (cache-hit probe)."""

    _revision = "fake-nli-rev-mb"
    device = "cpu"

    def __init__(self) -> None:
        self.batch_calls: List[int] = []

    def score_batch(
        self, *, pairs: List[Tuple[str, str]], batch_size: Optional[int] = None
    ) -> List[NliScore]:
        self.batch_calls.append(len(pairs))
        return MarkerStubNli().score_batch(pairs=pairs)


class _StopTriggerNli:
    """Writes the global stop sentinel on its FIRST call (between-blocks probe)."""

    _revision = "fake-nli-rev-mb"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def score_batch(
        self, *, pairs: List[Tuple[str, str]], batch_size: Optional[int] = None
    ) -> List[NliScore]:
        self.calls += 1
        if self.calls == 1:
            from lib.generation import stop_control

            stop_control.request_stop(scope="all", source="test")
        # Everything entailed → both blocks pass (isolate the stop behaviour).
        return [
            NliScore(entailment=0.95, neutral=0.03, contradiction=0.02)
            for _ in pairs
        ]


class _InlineExecutor:
    """Seam replacement for the spawn pool: runs the REAL worker code in-process.

    Calls the module-level ``_pool_worker_init`` once (exercising the factory
    import + one-NLI-per-worker contract), then runs each ``_pool_score_task``
    inline behind a resolved ``Future`` (so ``as_completed`` works unchanged).
    """

    last: Optional["_InlineExecutor"] = None

    def __init__(self, n_procs: int, factory_dotted: str) -> None:
        self.n_procs = n_procs
        self.factory_dotted = factory_dotted
        m._pool_worker_init(factory_dotted)  # one NLI load, like a real worker
        _InlineExecutor.last = self

    def submit(self, fn: Any, task: Any) -> "Future":
        fut: "Future" = Future()
        fut.set_result(fn(task))
        return fut

    def shutdown(self, wait: bool = True) -> None:  # noqa: D401 — pool parity
        return None


_CHUNK = (
    "The mitochondrion is the powerhouse of the cell and produces ATP "
    "through oxidative phosphorylation across the inner membrane."
)
_ENTAILED = (
    "<p>The mitochondrion is the cell's powerhouse and generates ATP "
    "through oxidative phosphorylation across its inner membrane [ENTAIL]. "
    "This energy production sustains the metabolic activity of the entire "
    "cell continuously [ENTAIL].</p>"
)
_FABRICATED = (
    "<p>The mitochondrion secretly manufactures cellular gold deep within "
    "its folded cristae structures every passing moment. It also quietly "
    "encodes ancient memories into the surrounding cytoplasm of the cell for "
    "safekeeping.</p>"
)
_CONTRADICTED = (
    "<p>The mitochondrion is the cell's powerhouse producing ATP through "
    "oxidative phosphorylation across its inner membrane [ENTAIL]. The "
    "mitochondrion actually destroys all cellular ATP and halts every "
    "metabolic process within the cell [CONTRA].</p>"
)
_SHORT_ENTAILED = (
    "<p>The mitochondrion generates ATP across its inner membrane [ENTAIL].</p>"
)


def _make_block(*, block_id: str, content: str) -> Block:
    return Block(
        block_id=block_id,
        block_type="concept",
        page_id=block_id.split("#")[0],
        sequence=0,
        content=content,
        source_ids=("semantik:slug#blk_0",),
        source_references=({"sourceId": "semantik:slug#blk_0"},),
    )


def _fixture_blocks() -> List[Block]:
    return [
        _make_block(block_id="p01#a", content=_ENTAILED),
        _make_block(block_id="p02#b", content=_FABRICATED),
        _make_block(block_id="p03#c", content=_CONTRADICTED),
        _make_block(block_id="p04#d", content=_SHORT_ENTAILED),
        _make_block(block_id="p05#e", content=_FABRICATED),
        _make_block(block_id="p06#f", content=_ENTAILED),
    ]


def _result_signature(result: Any) -> Tuple[Any, ...]:
    return (
        result.passed,
        result.action,
        result.score,
        tuple((i.code, i.severity, i.location) for i in result.issues),
    )


class _CaptureSpy:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch):
    for var in (
        _ENV_MICROBATCH_VALIDATORS,
        _ENV_VALIDATORS_PROCS,
        _ENV_NLI_VALIDATORS_FACTORY,
        _ENV_VALIDATION_CHECKPOINT,
        "ED4ALL_GENERATION_CHECKPOINT",
        "_PROSE_POOL_STUB_INITFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    m._WORKER_NLI = None
    _InlineExecutor.last = None
    yield
    m._WORKER_NLI = None


# --------------------------------------------------------------------- #
# Resolvers
# --------------------------------------------------------------------- #


def test_flag_resolver_parse_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _microbatch_validators_enabled() is False
    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, truthy)
        assert _microbatch_validators_enabled() is True
    for off in ("0", "false", "no", "off", "garbage", ""):
        monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, off)
        assert _microbatch_validators_enabled() is False


def test_procs_resolver_parse_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _resolve_validators_procs() == 4
    for raw, expect in (("8", 8), ("1", 1), ("0", 4), ("-5", 4), ("x", 4), ("", 4)):
        monkeypatch.setenv(_ENV_VALIDATORS_PROCS, raw)
        assert _resolve_validators_procs() == expect


# --------------------------------------------------------------------- #
# (1) Process-pool scoring — verdict identity vs serial (inline-executor seam)
# --------------------------------------------------------------------- #


def _run_serial() -> Any:
    return BlockProseEntailmentValidator(nli=MarkerStubNli()).validate(
        {"blocks": _fixture_blocks(), "source_chunks": {"semantik:slug#blk_0": _CHUNK}}
    )


def test_pool_verdict_identical_to_serial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off = _run_serial()

    monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, "1")
    monkeypatch.setenv(_ENV_NLI_VALIDATORS_FACTORY, _STUB_FACTORY_DOTTED)
    initfile = tmp_path / "inits.txt"
    monkeypatch.setenv("_PROSE_POOL_STUB_INITFILE", str(initfile))
    monkeypatch.setattr(m, "_make_scoring_pool", _InlineExecutor)

    on = BlockProseEntailmentValidator(nli=MarkerStubNli()).validate(
        {"blocks": _fixture_blocks(), "source_chunks": {"semantik:slug#blk_0": _CHUNK}}
    )

    assert _result_signature(on) == _result_signature(off)
    # The fixture is meaningful: it produced BOTH a failure and a pass.
    codes = {i.code for i in off.issues}
    assert _CODE_BLOCK_PROSE_UNGROUNDED in codes
    assert _CODE_BLOCK_PROSE_CONTRADICTED in codes
    assert off.passed is False
    # One NLI load per worker: the initializer ran exactly once for the pool.
    assert initfile.exists()
    assert len(initfile.read_text().splitlines()) == 1


def test_pool_decision_order_matches_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    off_cap = _CaptureSpy()
    BlockProseEntailmentValidator(nli=MarkerStubNli()).validate(
        {
            "blocks": _fixture_blocks(),
            "source_chunks": {"semantik:slug#blk_0": _CHUNK},
            "decision_capture": off_cap,
        }
    )

    monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, "1")
    monkeypatch.setenv(_ENV_NLI_VALIDATORS_FACTORY, _STUB_FACTORY_DOTTED)
    monkeypatch.setattr(m, "_make_scoring_pool", _InlineExecutor)
    on_cap = _CaptureSpy()
    BlockProseEntailmentValidator(nli=MarkerStubNli()).validate(
        {
            "blocks": _fixture_blocks(),
            "source_chunks": {"semantik:slug#blk_0": _CHUNK},
            "decision_capture": on_cap,
        }
    )

    def _ids(cap: _CaptureSpy) -> List[str]:
        ids: List[str] = []
        for evt in cap.events:
            rat = evt["rationale"]
            if "Block '" in rat:
                ids.append(rat.split("Block '")[1].split("'")[0])
        return ids

    assert _ids(on_cap) == _ids(off_cap)
    assert len(on_cap.events) == len(off_cap.events) == len(_fixture_blocks())


# --------------------------------------------------------------------- #
# (1b) Flag OFF → serial, pool never constructed
# --------------------------------------------------------------------- #


def test_flag_off_never_builds_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def _boom(*a: Any, **k: Any) -> Any:
        called["n"] += 1
        raise AssertionError("pool must not be built when the flag is off")

    monkeypatch.setattr(m, "_make_scoring_pool", _boom)
    nli = _CountingNli()
    validator = BlockProseEntailmentValidator(nli=nli)
    validator.validate(
        {"blocks": _fixture_blocks(), "source_chunks": {"semantik:slug#blk_0": _CHUNK}}
    )
    assert called["n"] == 0
    # Serial path: the injected nli was called directly (≥1 per scorable block).
    assert len(nli.batch_calls) >= len(_fixture_blocks())


# --------------------------------------------------------------------- #
# (1c) Pool-start failure → graceful serial fallback (verdict preserved)
# --------------------------------------------------------------------- #


def test_pool_failure_falls_back_to_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    off = _run_serial()

    from concurrent.futures.process import BrokenProcessPool

    def _broken(*a: Any, **k: Any) -> Any:
        raise BrokenProcessPool("simulated pool start failure")

    monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, "1")
    monkeypatch.setattr(m, "_make_scoring_pool", _broken)
    on = BlockProseEntailmentValidator(nli=MarkerStubNli()).validate(
        {"blocks": _fixture_blocks(), "source_chunks": {"semantik:slug#blk_0": _CHUNK}}
    )
    assert _result_signature(on) == _result_signature(off)


# --------------------------------------------------------------------- #
# (1d) Pickling roundtrip — the real cross-process risk
# --------------------------------------------------------------------- #


def test_pickling_roundtrip_task_factory_report() -> None:
    # Task tuple (idx, prose, cited_passages, ent_floor, con_floor).
    task = (3, "some prose text here", [{"chunk_id": "c1", "text": _CHUNK}], 0.70, 0.50)
    assert pickle.loads(pickle.dumps(task)) == task

    # The factory reference crosses the process boundary — must be picklable.
    f = pickle.loads(pickle.dumps(m.default_nli_factory))
    assert f is m.default_nli_factory

    # The returned GroundednessReport must pickle back to an identical shape.
    rep = score_groundedness(
        "The mitochondrion is the cell's powerhouse [ENTAIL].",
        [{"chunk_id": "c1", "text": _CHUNK}],
        nli=MarkerStubNli(),
        entailment_floor=0.70,
        contradiction_floor=0.50,
    )
    rep2 = pickle.loads(pickle.dumps(rep))
    assert rep2.to_dict() == rep.to_dict()

    # Round-tripping the worker task through the real task fn — init the worker
    # NLI via the stub factory dotted path (exercises _import_factory), then run
    # the real _pool_score_task (no GPU / no DeBERTa load).
    m._pool_worker_init(_STUB_FACTORY_DOTTED)
    assert isinstance(m._WORKER_NLI, MarkerStubNli)
    idx, worker_rep = m._pool_score_task(task)
    assert idx == 3
    assert worker_rep.available is True


# --------------------------------------------------------------------- #
# (2) Per-block resume sidecar — write / hit / invalidation
# --------------------------------------------------------------------- #


def _cache_records(cache_dir: Path) -> List[Dict[str, Any]]:
    p = cache_dir / m._CHECKPOINT_BASENAME
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def test_sidecar_write_then_hit_then_invalidate(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".prose_entailment_cache"
    inputs = {
        "blocks": _fixture_blocks(),
        "source_chunks": {"semantik:slug#blk_0": _CHUNK},
        "prose_entailment_cache_dir": str(cache_dir),
    }

    # First run — writes a sidecar record per scorable block.
    nli1 = _CountingNli()
    r1 = BlockProseEntailmentValidator(nli=nli1).validate(dict(inputs))
    recs = _cache_records(cache_dir)
    assert {rec["unit_id"] for rec in recs} == {b.block_id for b in _fixture_blocks()}
    assert len(nli1.batch_calls) >= len(_fixture_blocks())

    # Second run — every block is a cache HIT; the NLI is NEVER called.
    nli2 = _CountingNli()
    r2 = BlockProseEntailmentValidator(nli=nli2).validate(dict(inputs))
    assert nli2.batch_calls == []
    assert _result_signature(r2) == _result_signature(r1)

    # Invalidation — changing one block's prose moves its fingerprint → MISS
    # for that block (others still hit). The NLI is called again (for the miss).
    changed = _fixture_blocks()
    changed[0] = _make_block(
        block_id="p01#a",
        content="<p>Entirely different grounded prose about ATP synthesis "
        "on the inner membrane of the mitochondrion here [ENTAIL].</p>",
    )
    nli3 = _CountingNli()
    BlockProseEntailmentValidator(nli=nli3).validate(
        {
            "blocks": changed,
            "source_chunks": {"semantik:slug#blk_0": _CHUNK},
            "prose_entailment_cache_dir": str(cache_dir),
        }
    )
    assert nli3.batch_calls != []  # the changed block was re-scored


def test_sidecar_disabled_when_flag_falsey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VALIDATION_CHECKPOINT, "off")
    cache_dir = tmp_path / ".prose_entailment_cache"
    BlockProseEntailmentValidator(nli=_CountingNli()).validate(
        {
            "blocks": _fixture_blocks(),
            "source_chunks": {"semantik:slug#blk_0": _CHUNK},
            "prose_entailment_cache_dir": str(cache_dir),
        }
    )
    # Flag off → no cache dir/file is created (byte-identical serial legacy).
    assert not (cache_dir / m._CHECKPOINT_BASENAME).exists()


def test_never_caches_degraded_report(tmp_path: Path) -> None:
    """A report with ``available=False`` must never be persisted."""

    class _UnavailableNli:
        _revision = "x"
        device = "cpu"

        def score_batch(self, *, pairs: Any, batch_size: Any = None) -> Any:
            raise AssertionError("should not be reached — resolves unavailable")

    # score_groundedness treats a None-resolving nli as unavailable; simulate by
    # passing an nli that the validator accepts but whose report is unavailable.
    # Easiest: monkeypatch score_groundedness at the module to return unavailable.
    import lib.retrieval.groundedness as g

    from lib.retrieval.groundedness import GroundednessReport

    cache_dir = tmp_path / ".prose_entailment_cache"
    orig = m.score_groundedness
    try:
        m.score_groundedness = lambda *a, **k: GroundednessReport(  # type: ignore
            available=False, reason="stub"
        )
        # available=False → the validator returns the NLI_DEPS_MISSING degrade,
        # and nothing is cached.
        BlockProseEntailmentValidator(nli=MarkerStubNli()).validate(
            {
                "blocks": _fixture_blocks()[:1],
                "source_chunks": {"semantik:slug#blk_0": _CHUNK},
                "prose_entailment_cache_dir": str(cache_dir),
            }
        )
    finally:
        m.score_groundedness = orig
    assert _cache_records(cache_dir) == []
    _ = (g, _UnavailableNli)  # silence unused


# --------------------------------------------------------------------- #
# (3) Stop-sentinel — pauses BETWEEN blocks, persists completed work
# --------------------------------------------------------------------- #


def test_stop_sentinel_pauses_between_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lib.generation import stop_control
    from lib.generation.stop_control import GracefulStopRequested

    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "state_runs"))
    stop_control.clear_stop(include_global=True)

    cache_dir = tmp_path / ".prose_entailment_cache"
    blocks = [
        _make_block(block_id="s01#a", content=_ENTAILED),
        _make_block(block_id="s02#b", content=_ENTAILED),
    ]
    nli = _StopTriggerNli()  # writes the sentinel while scoring block 0
    try:
        with pytest.raises(GracefulStopRequested) as ei:
            BlockProseEntailmentValidator(nli=nli).validate(
                {
                    "blocks": blocks,
                    "source_chunks": {"semantik:slug#blk_0": _CHUNK},
                    "prose_entailment_cache_dir": str(cache_dir),
                }
            )
        # Block 0 finished + checkpointed; block 1 never scored.
        assert ei.value.units_completed == 1
        recs = _cache_records(cache_dir)
        assert {r["unit_id"] for r in recs} == {"s01#a"}
    finally:
        stop_control.clear_stop(include_global=True)


# --------------------------------------------------------------------- #
# (1e) Real spawn-pool smoke test — end-to-end through actual processes.
#      Verdict identity is the hard assertion; a spawn failure falls back to
#      serial (same verdict), so this never flakes.
# --------------------------------------------------------------------- #


def test_real_spawn_pool_verdict_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off = _run_serial()

    monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, "1")
    monkeypatch.setenv(_ENV_VALIDATORS_PROCS, "2")
    monkeypatch.setenv(_ENV_NLI_VALIDATORS_FACTORY, _STUB_FACTORY_DOTTED)
    initfile = tmp_path / "spawn_inits.txt"
    monkeypatch.setenv("_PROSE_POOL_STUB_INITFILE", str(initfile))

    on = BlockProseEntailmentValidator(nli=MarkerStubNli()).validate(
        {"blocks": _fixture_blocks(), "source_chunks": {"semantik:slug#blk_0": _CHUNK}}
    )
    assert _result_signature(on) == _result_signature(off)

    # Best-effort cross-process proof: if the real pool ran, workers (distinct
    # PIDs) recorded their init. A spawn failure falls back to serial (empty
    # file) without failing the verdict assertion above.
    if initfile.exists():
        pids = {int(x) for x in initfile.read_text().split()}
        assert all(pid != os.getpid() for pid in pids) or pids == set()
