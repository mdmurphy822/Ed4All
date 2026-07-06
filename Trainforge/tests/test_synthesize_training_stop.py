"""Graceful-stop ("checkpoint on command") regression tests for
``Trainforge/synthesize_training.py::run_synthesis``.

These extend the resume-checkpoint suite in
``test_synthesize_training.py:1795-2129`` (``preserved on budget-exceeded``
at :1923 is the structural template). Where the budget path BREAKS out of
the chunk loop and then RETURNS a ``SynthesisStats`` normally — which the
caller reads as phase COMPLETE (``capped_at_max_dispatches`` is a
success-shaped return) — the graceful-stop path must instead RAISE
``GracefulStopRequested`` at the same unit boundary so the executor maps it
to a PAUSED (never failed, never retried) result and the phase stays
resumable. A completed phase would never be re-dispatched on ``--resume``,
so the un-synthesized tail would be silently dropped; raising is therefore
mandatory here, not a mirror of the budget break.

Every test asserts the mechanical guarantee: the paraphrase/synthesis path
arms the stop sentinel after call N -> ``GracefulStopRequested`` is raised,
the resume sidecar holds EXACTLY N records, and the synthesis path recorded
EXACTLY N calls (no work leaks past the observed stop). The resume leg
clears the sentinel, re-runs, and completes the remaining total-minus-N
calls with byte-equivalent output modulo the per-run ``decision_capture_id``
field. A pre-armed sentinel yields zero calls.

All tests are fully offline + deterministic: ``provider="mock"`` (the
deterministic template factory), the four per-pair grounding validators
neutralized so every synthesized pair is accepted (these tests exercise the
STOP mechanism, NOT grounding calibration — that is covered elsewhere), and
the stop sentinel written into an isolated ``ED4ALL_STATE_RUNS_DIR`` via the
``state_runs_isolated`` fixture so a test never touches the real
``state/runs/`` of a live run (risk R1 / R2 in the plan).
"""

from __future__ import annotations

import json as _json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Trainforge.synthesize_training as synthesize_training  # noqa: E402
from Trainforge.synthesize_training import run_synthesis  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402


FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)
CHECKPOINT_NAME = ".synthesis_pairs_checkpoint.jsonl"
_VOLATILE_PAIR_FIELDS = ("decision_capture_id",)


def _make_working_copy(tmp_path: Path, name: str = "mc") -> Path:
    """Copy the read-only fixture into tmp so ``run_synthesis`` can write.

    Strips any stale canonical artifacts + resume sidecar so each run starts
    from a clean training_specs dir.
    """
    dst = tmp_path / name
    shutil.copytree(FIXTURE_ROOT, dst)
    ts = dst / "training_specs"
    for stale in (
        ts / "instruction_pairs.jsonl",
        ts / "preference_pairs.jsonl",
        ts / "instruction_pairs.jsonl.in_progress",
        ts / "preference_pairs.jsonl.in_progress",
        ts / CHECKPOINT_NAME,
    ):
        stale.unlink(missing_ok=True)
    return dst


def _neutralize_pair_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every per-pair grounding validator to ``"validated"``.

    The four filters (W2.E promotion, W4.A claim-support, W4.B LO-refs, W4.C
    objective-delivery) reject ungrounded mock/template content, which would
    make provider-calls and sidecar-records diverge (a dispatched-but-rejected
    pair is never checkpointed). These STOP tests care only about the loop's
    stop disposition, so we pin every pair to accepted -> each synthesis call
    yields exactly one checkpointed record.
    """
    import lib.validators.pair.claim_support as pcs
    import lib.validators.pair.lo_refs as plr
    import lib.validators.pair.objective_delivery as pod
    import lib.validators.pair.promotion as tpp

    def _validated(*_a: Any, **_k: Any):
        return ("validated", None, {})

    monkeypatch.setattr(
        tpp.TrainingPairPromotionValidator, "validate_pair", _validated,
    )
    monkeypatch.setattr(
        pcs.PairClaimSupportValidator, "validate_pair", _validated,
    )
    monkeypatch.setattr(
        plr.PairLearningOutcomeRefsValidator, "validate_pair", _validated,
    )
    monkeypatch.setattr(
        pod.PairObjectiveDeliveryValidator, "validate_pair", _validated,
    )


class _CallCounter:
    """Wraps the two synthesis entry points to count calls and, optionally,
    arm the global stop sentinel exactly once when the Nth call fires.

    ``arm_after`` counts SYNTHESIS calls (instruction + preference). The
    fixture emits exactly 2 calls per eligible chunk (1 instruction variant +
    1 preference), so an even ``arm_after`` lands on a chunk boundary. The
    stop is observed at the NEXT loop top, so the current chunk always
    finishes and is fully checkpointed before the raise — meaning
    ``calls == records`` for ANY arm point (nothing dispatches after the
    sentinel is seen at a loop top).
    """

    def __init__(self, arm_after: Optional[int] = None) -> None:
        self.calls = 0
        self._arm_after = arm_after

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_inst = synthesize_training.synthesize_instruction_pair
        real_pref = synthesize_training.synthesize_preference_pair

        def _wrap(real: Callable[..., Any]) -> Callable[..., Any]:
            def _counted(*args: Any, **kwargs: Any):
                self.calls += 1
                if self._arm_after is not None and self.calls == self._arm_after:
                    stop_control.request_stop(
                        scope="all", reason="test_stop", source="unit-test",
                    )
                return real(*args, **kwargs)

            return _counted

        monkeypatch.setattr(
            synthesize_training, "synthesize_instruction_pair", _wrap(real_inst),
        )
        monkeypatch.setattr(
            synthesize_training, "synthesize_preference_pair", _wrap(real_pref),
        )


def _read_checkpoint(working: Path) -> List[dict]:
    cp = working / "training_specs" / CHECKPOINT_NAME
    if not cp.exists():
        return []
    return [
        _json.loads(line)
        for line in cp.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_pairs(working: Path, name: str) -> List[dict]:
    p = working / "training_specs" / name
    return [
        _json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_volatile(pairs: List[dict]) -> str:
    """Canonical JSON of pairs sorted by identity, minus per-run fields."""
    cleaned = [
        {k: v for k, v in r.items() if k not in _VOLATILE_PAIR_FIELDS}
        for r in pairs
    ]
    cleaned.sort(key=lambda r: (r.get("chunk_id", ""), r.get("prompt", "")))
    return _json.dumps(cleaned, sort_keys=True)


@pytest.fixture(autouse=True)
def _clear_global_sentinel(state_runs_isolated):
    """Every test in this module runs under an isolated ``state/runs`` and
    clears the operator-owned global ``STOP_ALL`` on teardown so a mid-test
    arm never leaks into a sibling test.
    """
    yield
    stop_control.clear_stop(include_global=True)


# --------------------------------------------------------------------------- #
# Leg 1 — stop after N pairs -> sidecar exactly N records, N calls, RAISED.
# --------------------------------------------------------------------------- #
def test_stop_after_n_pairs_checkpoints_exactly_n_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm the sentinel after call N (a chunk boundary) -> the loop raises
    ``GracefulStopRequested`` at the next loop top, the resume sidecar holds
    EXACTLY N records, and the synthesis path recorded EXACTLY N calls.

    Also proves the phase is NOT marked complete on stop: the run RAISES
    (never returns a ``SynthesisStats``), the resume sidecar is PRESERVED
    (a clean/complete run unlinks it), and NO final canonical artifact was
    written (the raise fires before the post-loop ``open("w")`` persist —
    risk R5).
    """
    _neutralize_pair_validators(monkeypatch)
    working = _make_working_copy(tmp_path)
    n = 6  # 3 eligible chunks x (1 instruction + 1 preference)
    counter = _CallCounter(arm_after=n)
    counter.install(monkeypatch)

    with pytest.raises(GracefulStopRequested) as excinfo:
        run_synthesis(
            corpus_dir=working,
            course_code="MINI_TRAINING_101",
            provider="mock",
            seed=17,
        )

    # Provider recorded EXACTLY N calls — nothing dispatched after the
    # sentinel was observed at the loop top.
    assert counter.calls == n
    # Resume sidecar holds EXACTLY N records (one per accepted pair).
    records = _read_checkpoint(working)
    assert len(records) == n
    # The exception payload reports the completed-unit count truthfully.
    assert excinfo.value.units_completed == n
    assert excinfo.value.site_id == "training_synthesis.pair_loop"

    # Phase-not-complete evidence: sidecar preserved (clean-exit unlink
    # never ran) + no final artifact written (raised before persist).
    cp = working / "training_specs" / CHECKPOINT_NAME
    assert cp.exists()
    assert not (working / "training_specs" / "instruction_pairs.jsonl").exists()
    assert not (working / "training_specs" / "preference_pairs.jsonl").exists()


# --------------------------------------------------------------------------- #
# Leg 2 — resume: clear_stop, re-run -> total-N calls, byte-equivalent output.
# --------------------------------------------------------------------------- #
def test_resume_after_stop_completes_remainder_byte_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopped run, once the sentinel is cleared, resumes from the sidecar:
    it re-synthesizes only the total-minus-N remaining pairs (the N cached
    pairs skip the LLM dispatch entirely) and lands output byte-equivalent to
    a single uninterrupted run modulo the per-run ``decision_capture_id``.
    """
    _neutralize_pair_validators(monkeypatch)

    # --- Reference: one uninterrupted run to capture the full call count and
    #     the canonical artifact bytes. ---
    baseline = _make_working_copy(tmp_path, name="baseline")
    base_counter = _CallCounter()
    base_counter.install(monkeypatch)
    run_synthesis(
        corpus_dir=baseline,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )
    full_total = base_counter.calls
    ref_inst = _strip_volatile(_load_pairs(baseline, "instruction_pairs.jsonl"))
    ref_pref = _strip_volatile(_load_pairs(baseline, "preference_pairs.jsonl"))
    assert full_total > 6  # sanity: the stop below is genuinely partial

    # --- Stop leg: arm after N=6 -> raise, checkpoint holds 6. ---
    working = _make_working_copy(tmp_path, name="resumed")
    n = 6
    stop_counter = _CallCounter(arm_after=n)
    stop_counter.install(monkeypatch)
    with pytest.raises(GracefulStopRequested):
        run_synthesis(
            corpus_dir=working,
            course_code="MINI_TRAINING_101",
            provider="mock",
            seed=17,
        )
    assert stop_counter.calls == n
    assert len(_read_checkpoint(working)) == n

    # --- Resume leg: clear the sentinel, re-run on the SAME dir. ---
    stop_control.clear_stop(include_global=True)
    resume_counter = _CallCounter()
    resume_counter.install(monkeypatch)
    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    # Only the remaining total-minus-N pairs re-dispatched; the N cached
    # pairs short-circuited past the synthesis path.
    assert resume_counter.calls == full_total - n
    # Output byte-equivalent to the uninterrupted run (modulo reuse counters).
    assert _strip_volatile(_load_pairs(working, "instruction_pairs.jsonl")) == ref_inst
    assert _strip_volatile(_load_pairs(working, "preference_pairs.jsonl")) == ref_pref
    # Clean resume unlinks the sidecar — the phase is now genuinely complete.
    assert not (working / "training_specs" / CHECKPOINT_NAME).exists()


# --------------------------------------------------------------------------- #
# Leg 3 — pre-armed sentinel -> zero calls.
# --------------------------------------------------------------------------- #
def test_pre_armed_sentinel_yields_zero_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sentinel that already exists when the run starts stops the loop at
    the very first unit boundary: zero synthesis calls, zero checkpointed
    records, ``GracefulStopRequested`` raised with ``units_completed == 0``.
    """
    _neutralize_pair_validators(monkeypatch)
    working = _make_working_copy(tmp_path)
    counter = _CallCounter()  # never self-arms; sentinel pre-armed below
    counter.install(monkeypatch)

    stop_control.request_stop(scope="all", reason="pre_armed", source="unit-test")

    with pytest.raises(GracefulStopRequested) as excinfo:
        run_synthesis(
            corpus_dir=working,
            course_code="MINI_TRAINING_101",
            provider="mock",
            seed=17,
        )

    assert counter.calls == 0
    assert excinfo.value.units_completed == 0
    assert len(_read_checkpoint(working)) == 0
    assert not (working / "training_specs" / "instruction_pairs.jsonl").exists()


# --------------------------------------------------------------------------- #
# Phase-not-complete contrast — stop RAISES where a clean run RETURNS.
# --------------------------------------------------------------------------- #
def test_stop_raises_where_clean_run_returns_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean (unarmed) run RETURNS a ``SynthesisStats`` and unlinks the
    sidecar (phase complete). The identical run with the sentinel armed at a
    chunk boundary RAISES ``GracefulStopRequested`` and PRESERVES the sidecar
    instead — the disposition that keeps a stopped phase resumable rather than
    silently completing (unlike the budget-exhausted break, which returns a
    success-shaped ``capped_at_max_dispatches`` stats object).
    """
    _neutralize_pair_validators(monkeypatch)

    # Clean run returns normally and deletes the sidecar.
    clean = _make_working_copy(tmp_path, name="clean")
    _CallCounter().install(monkeypatch)
    stats = run_synthesis(
        corpus_dir=clean,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )
    assert stats is not None
    assert stats.capped_at_max_dispatches is False
    assert not (clean / "training_specs" / CHECKPOINT_NAME).exists()

    # Armed run raises and preserves the sidecar (not marked complete).
    stopped = _make_working_copy(tmp_path, name="stopped")
    _CallCounter(arm_after=4).install(monkeypatch)
    with pytest.raises(GracefulStopRequested):
        run_synthesis(
            corpus_dir=stopped,
            course_code="MINI_TRAINING_101",
            provider="mock",
            seed=17,
        )
    assert (stopped / "training_specs" / CHECKPOINT_NAME).exists()
