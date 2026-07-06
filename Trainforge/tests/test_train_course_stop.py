"""trainforge_train graceful-stop seam (P4b / AMENDMENT #3).

The trainer loop is TRL's ``SFTTrainer`` / ``DPOTrainer`` — HF ``transformers``
``Trainer`` subclasses whose per-step / per-epoch callback boundary is the
graceful-stop seam. ``Trainforge/training/peft_trainer.py`` wires a
``TrainerCallback`` (built by ``_build_stop_callback``) that, when the SAME
``lib.generation.stop_control`` sentinel every other Ed4All stage checks is
armed, asks the trainer to (1) flush its NATIVE ``checkpoint-<step>`` resume
dir and (2) stop cleanly; the pause is then surfaced as
``GracefulStopRequested`` (``_raise_if_stopped``) so the runner/executor mark
the phase ``paused`` — never ``completed``.

GPU caveat (per the plan): real QLoRA training can't run on CI, so these tests
exercise the seam with a CPU-only fake trainer that mirrors HF's inner loop
(fire ``on_step_end`` per step, honor ``control.should_save`` / ``.should_
training_stop``). No torch / transformers / GPU. The reaction logic lives in
``_GracefulStopMixin`` (no ``transformers`` import) so the test callback mixes
it in directly.

The three stop legs mapped onto trainer semantics:

- **stop after N** — sentinel armed after step N → exactly ONE native
  checkpoint flushed at step N+1's boundary, training halts there, the surface
  raises with ``units_completed == N+1``.
- **resume** — a resumed run finds the paused run's ``checkpoint-*`` and
  ``_resume_arg`` returns ``True`` (``Trainer.train(resume_from_checkpoint=True)``
  auto-selects the latest), so training continues from the native checkpoint.
- **pre-armed** — the ``fit_sft`` / ``fit_dpo`` preflight ``check_stop`` raises
  before any weight load (zero GPU work); a sentinel armed before the first
  step stops the loop at step 1.

Hermetic: tmp ``ED4ALL_STATE_RUNS_DIR`` + a synthetic ``ED4ALL_RUN_ID``; no
course slugs / paths.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from Trainforge.training import compute_backend as cbmod  # noqa: E402
from Trainforge.training import peft_trainer as pt  # noqa: E402
from Trainforge.training.peft_trainer import _GracefulStopMixin  # noqa: E402


_RUN_ID = "STOP_TRAIN_TESTRUN"


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #
@pytest.fixture()
def stop_env(tmp_path, monkeypatch):
    """Per-test sentinel isolation: tmp state/runs + a synthetic run_id."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    stop_control.clear_stop(include_global=True)
    yield runs
    stop_control.clear_stop(include_global=True)


# --------------------------------------------------------------------------- #
# CPU-only fakes mirroring HF Trainer's callback contract
# --------------------------------------------------------------------------- #
class _FakeControl:
    """Stand-in for ``transformers.TrainerControl`` (only the flags we set)."""

    def __init__(self) -> None:
        self.should_save = False
        self.should_training_stop = False


class _FakeState:
    """Stand-in for ``transformers.TrainerState`` (only ``global_step``)."""

    def __init__(self) -> None:
        self.global_step = 0


class _StopCallback(_GracefulStopMixin):
    """Test callback: the real reaction mixin WITHOUT the transformers base.

    ``min_interval_s=0.0`` disables the ``StopPoller`` throttle so the sentinel
    is re-probed on every boundary (deterministic under the fake loop).
    """

    def __init__(self) -> None:
        self._init_stop(min_interval_s=0.0)


class _ArmAtStep:
    """Fake callback that arms the run-scoped sentinel once ``global_step`` hits
    ``at_step`` — models an operator ``ed4all stop`` landing mid-training."""

    def __init__(self, at_step: int) -> None:
        self.at_step = at_step

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
        if state.global_step == self.at_step:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
        return control


class _FakeTrainer:
    """Minimal HF-``Trainer``-shaped loop that drives registered callbacks.

    Tight enough to exercise the graceful-stop callback contract on CPU: each
    step bumps ``global_step`` and fires ``on_step_end`` for every callback (in
    registration order), then honors ``control.should_save`` (flush a native
    ``checkpoint-<step>`` dir + reset the flag) and ``control.should_training_
    stop`` (break). No torch / transformers.
    """

    def __init__(self, output_dir: Path, total_steps: int = 5) -> None:
        self.output_dir = Path(output_dir)
        self.total_steps = total_steps
        self.state = _FakeState()
        self.control = _FakeControl()
        self._callbacks: List[Any] = []
        self.saved_checkpoints: List[Path] = []
        self.resume_from_checkpoint: Any = "UNSET"

    def add_callback(self, cb: Any) -> None:
        self._callbacks.append(cb)

    def _fire(self, event: str) -> None:
        for cb in self._callbacks:
            getattr(cb, event)(None, self.state, self.control)

    def _save_checkpoint(self) -> None:
        ckpt = self.output_dir / f"checkpoint-{self.state.global_step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "trainer_state.json").write_text("{}", encoding="utf-8")
        self.saved_checkpoints.append(ckpt)

    def train(self, resume_from_checkpoint: Any = None) -> None:
        self.resume_from_checkpoint = resume_from_checkpoint
        for _ in range(self.total_steps):
            self.state.global_step += 1
            self._fire("on_step_end")
            if self.control.should_save:
                self._save_checkpoint()
                self.control.should_save = False
            if self.control.should_training_stop:
                break


# --------------------------------------------------------------------------- #
# Callback reaction unit contract
# --------------------------------------------------------------------------- #
def test_callback_noop_when_no_sentinel(stop_env):
    cb = _StopCallback()
    ctrl, state = _FakeControl(), _FakeState()
    state.global_step = 7
    cb.on_step_end(None, state, ctrl)
    assert ctrl.should_save is False
    assert ctrl.should_training_stop is False
    assert cb.stop_triggered is False
    # Progress is tracked for the resume hint even on the no-stop path.
    assert cb.stop_global_step == 7


def test_callback_triggers_on_step_end_when_armed(stop_env):
    cb = _StopCallback()
    stop_control.request_stop(scope="run", reason="test", source="test")
    ctrl, state = _FakeControl(), _FakeState()
    state.global_step = 3
    cb.on_step_end(None, state, ctrl)
    assert ctrl.should_save is True  # flush the native checkpoint
    assert ctrl.should_training_stop is True  # then halt cleanly
    assert cb.stop_triggered is True
    assert cb.stop_global_step == 3


def test_callback_triggers_on_epoch_end_when_armed(stop_env):
    cb = _StopCallback()
    stop_control.request_stop(scope="run", reason="test", source="test")
    ctrl, state = _FakeControl(), _FakeState()
    state.global_step = 12
    cb.on_epoch_end(None, state, ctrl)
    assert ctrl.should_save is True
    assert ctrl.should_training_stop is True
    assert cb.stop_triggered is True


def test_callback_latches_after_trigger(stop_env):
    """Once stopped, a fresh control on a later boundary is NOT re-flagged."""
    cb = _StopCallback()
    stop_control.request_stop(scope="run", reason="test", source="test")
    cb.on_step_end(None, _FakeState(), _FakeControl())
    assert cb.stop_triggered is True
    # Clear the sentinel + hand a fresh control: the latch short-circuits.
    stop_control.clear_stop(include_global=True)
    fresh = _FakeControl()
    cb.on_step_end(None, _FakeState(), fresh)
    assert fresh.should_save is False
    assert fresh.should_training_stop is False
    assert cb.stop_triggered is True  # stays latched


# --------------------------------------------------------------------------- #
# Pause-surface helper
# --------------------------------------------------------------------------- #
def test_raise_if_stopped_raises_with_units(stop_env):
    cb = _StopCallback()
    cb.stop_triggered = True
    cb.stop_global_step = 42
    with pytest.raises(GracefulStopRequested) as ei:
        pt._raise_if_stopped(cb, "trainforge_train.fit_sft")
    assert ei.value.units_completed == 42
    assert ei.value.site_id == "trainforge_train.fit_sft"


def test_raise_if_stopped_noop_when_not_triggered(stop_env):
    cb = _StopCallback()
    pt._raise_if_stopped(cb, "trainforge_train.fit_sft")  # no raise
    pt._raise_if_stopped(None, "trainforge_train.fit_sft")  # None-safe


# --------------------------------------------------------------------------- #
# Preflight (pre-armed → zero trainer work)
# --------------------------------------------------------------------------- #
def test_preflight_check_stop_raises_when_pre_armed(stop_env):
    """A sentinel armed before ``fit_sft`` runs stops at the preflight
    ``check_stop`` — before any weight load (the pre-armed → 0-work leg)."""
    stop_control.request_stop(scope="run", reason="test", source="test")
    with pytest.raises(GracefulStopRequested):
        pt.check_stop("trainforge_train.fit_sft.preflight", 0)


# --------------------------------------------------------------------------- #
# Fake-trainer integration — the three stop legs against the callback contract
# --------------------------------------------------------------------------- #
def test_fake_trainer_stops_after_n_and_checkpoints(stop_env):
    """Stop-after-N: sentinel armed after step 2 → one native checkpoint at
    step 3, training halts, surface raises with ``units_completed == 3``."""
    trainer = _FakeTrainer(stop_env / "run_dir", total_steps=5)
    stop_cb = _StopCallback()
    # stop_cb FIRST so at step 2 it probes BEFORE _ArmAtStep arms the sentinel
    # (arming takes effect at the NEXT boundary, step 3).
    trainer.add_callback(stop_cb)
    trainer.add_callback(_ArmAtStep(at_step=2))

    trainer.train(resume_from_checkpoint=pt._resume_arg(trainer.output_dir))

    assert trainer.resume_from_checkpoint is None  # fresh run, no prior ckpt
    assert stop_cb.stop_triggered is True
    assert trainer.state.global_step == 3  # halted at step 3, not 5
    assert [c.name for c in trainer.saved_checkpoints] == ["checkpoint-3"]
    with pytest.raises(GracefulStopRequested) as ei:
        pt._raise_if_stopped(stop_cb, "trainforge_train.fit_sft")
    assert ei.value.units_completed == 3


def test_fake_trainer_pre_armed_stops_at_first_step(stop_env):
    """Pre-armed at the loop level → stop at step 1, exactly one checkpoint."""
    stop_control.request_stop(scope="run", reason="test", source="test")
    trainer = _FakeTrainer(stop_env / "run_dir", total_steps=5)
    stop_cb = _StopCallback()
    trainer.add_callback(stop_cb)

    trainer.train()

    assert stop_cb.stop_triggered is True
    assert trainer.state.global_step == 1
    assert [c.name for c in trainer.saved_checkpoints] == ["checkpoint-1"]


def test_fake_trainer_runs_to_completion_when_unarmed(stop_env):
    """No sentinel → callback is a no-op: all steps run, no stop checkpoint,
    no pause surfaced."""
    trainer = _FakeTrainer(stop_env / "run_dir", total_steps=4)
    stop_cb = _StopCallback()
    trainer.add_callback(stop_cb)

    trainer.train()

    assert stop_cb.stop_triggered is False
    assert trainer.saved_checkpoints == []
    assert trainer.state.global_step == 4
    pt._raise_if_stopped(stop_cb, "trainforge_train.fit_sft")  # no raise


# --------------------------------------------------------------------------- #
# Resume detection (native-checkpoint restore path)
# --------------------------------------------------------------------------- #
def test_resume_arg_detects_native_checkpoint(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    assert pt._has_trainer_checkpoint(d) is False
    assert pt._resume_arg(d) is None
    (d / "checkpoint-5").mkdir()
    assert pt._has_trainer_checkpoint(d) is True
    # True → HF Trainer auto-selects the latest checkpoint-* on resume.
    assert pt._resume_arg(d) is True
    # Missing dir degrades best-effort to "fresh".
    assert pt._has_trainer_checkpoint(tmp_path / "does-not-exist") is False


def test_resume_after_stop_continues_from_checkpoint(stop_env):
    """End-to-end resume leg on the fake loop: a paused run leaves a
    ``checkpoint-*``; the resumed run detects it and passes
    ``resume_from_checkpoint=True`` to ``train``."""
    run_dir = stop_env / "run_dir"

    # Interrupted leg: stop after step 2 → checkpoint-3 on disk.
    interrupted = _FakeTrainer(run_dir, total_steps=5)
    icb = _StopCallback()
    interrupted.add_callback(icb)
    interrupted.add_callback(_ArmAtStep(at_step=2))
    interrupted.train(resume_from_checkpoint=pt._resume_arg(run_dir))
    assert icb.stop_triggered is True
    assert (run_dir / "checkpoint-3").is_dir()

    # Resume leg: sentinel cleared, same run_dir → native checkpoint detected.
    stop_control.clear_stop(include_global=True)
    resumed = _FakeTrainer(run_dir, total_steps=5)
    resumed.add_callback(_StopCallback())
    resumed.train(resume_from_checkpoint=pt._resume_arg(run_dir))
    assert resumed.resume_from_checkpoint is True  # restores from checkpoint-3
    assert resumed.state.global_step == 5  # runs to completion this time


# --------------------------------------------------------------------------- #
# compute_backend carve-out — a DPO-phase stop is a PAUSE, not a DPO failure
# --------------------------------------------------------------------------- #
def test_local_backend_dpo_stop_not_swallowed(tmp_path, monkeypatch, stop_env):
    """A ``GracefulStopRequested`` raised inside ``fit_dpo`` must propagate as a
    pause — never be caught by LocalBackend.run's ``except Exception`` SFT-
    fallback / dpo_fail_hard arm."""
    instr = tmp_path / "instruction_pairs.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    instr.write_text('{"prompt": "p", "completion": "c"}\n', encoding="utf-8")
    pref.write_text(
        '{"prompt": "p", "chosen": "a", "rejected": "b"}\n', encoding="utf-8"
    )
    out_dir = tmp_path / "run_dir"

    class _FakePEFT:
        def __init__(self, *, base_model, training_config) -> None:
            pass

        def fit_sft(self, pairs, output_dir):
            (Path(output_dir) / "adapter_model.safetensors").write_text("w")
            return Path(output_dir) / "adapter_model.safetensors"

        def fit_dpo(self, pairs, sft_adapter_path, output_dir):
            raise GracefulStopRequested("trainforge_train.fit_dpo", 5)

    # LocalBackend.run does ``from ...peft_trainer import PEFTTrainer`` locally,
    # so patch the source attribute the import resolves.
    monkeypatch.setattr(pt, "PEFTTrainer", _FakePEFT)

    backend = cbmod.LocalBackend(allow_no_gpu=True)
    spec = cbmod.TrainingJobSpec(
        course_slug="unit-test",
        base_model="qwen2.5-1.5b",
        instruction_pairs_path=instr,
        preference_pairs_path=pref,
        training_config={
            "min_dpo_pairs": 1,
            "dpo_preference_filter": "all",
            "dpo_fail_hard": True,
        },
        output_dir=out_dir,
        run_dpo=True,
    )

    with pytest.raises(GracefulStopRequested):
        backend.run(spec)
