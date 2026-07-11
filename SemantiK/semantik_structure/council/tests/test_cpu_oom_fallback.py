"""Regression — council CUDA-OOM CPU fallback (tech-debt D14).

The GPU-contention anti-silent-degradation guard. When another process pins
the whole GPU (a vLLM-served LLM holding all VRAM), an in-process council
backbone forward raises a CUDA out-of-memory error. The historic behavior
caught that in ``orchestrator._safe_run`` and SILENTLY SKIPPED the head — the
run reported ``gates=pass`` while losing e.g. merge_or_split heading
refinement on a chapter. The fix (``runner.run_bert``) instead:

    (a) retries the forward on CPU (refinement preserved, not skipped),
    (b) returns the REAL (non-empty, non-None) result, and
    (c) LATCHES to CPU so a subsequent forward goes straight to CPU rather
        than re-attempting CUDA and re-OOMing on every span-batch.

No real GPU is needed: the CUDA OOM is simulated by a patched runner callable
that raises a CUDA-OOM-shaped error while the (fake) backbone reports a
``cuda`` device, and returns a real ``BertOutput`` once the backbone has been
moved to CPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from semantik_structure.council import runner as _runner  # noqa: E402
from semantik_structure.council.types import BertOutput, TypedSignal  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes — a backbone/adapter that need no model weights or GPU.
# ---------------------------------------------------------------------------


class _FakeBackbone:
    def __init__(self) -> None:
        self.device = "cuda"
        self.name = "fake/backbone"
        self.revision = "main"
        self.force_to_cpu_calls = 0

    def force_to_cpu(self) -> None:
        self.force_to_cpu_calls += 1
        self.device = "cpu"


class _FakeAdapter:
    def __init__(self, backbone, spec) -> None:  # noqa: ANN001
        self.backbone = backbone
        self.spec = spec
        self.peft_model = object()
        self.load_calls = 0

    def load(self):  # noqa: ANN201
        self.load_calls += 1
        return self


def _real_output() -> BertOutput:
    """A non-empty BertOutput — the "refinement was preserved" evidence."""
    return BertOutput(
        bert_name="merge_or_split",
        signals=[
            TypedSignal(
                head_name="same_or_not",
                region_id=0,
                top_k_labels=["same"],
                top_k_confidences=[0.99],
                feature_provenance={},
            )
        ],
        backbone_version="fake/backbone@main",
        adapter_version="merge_or_split@fake",
    )


@pytest.fixture()
def _wire(monkeypatch):
    """Wire a synthetic council BERT into the runner with fake plumbing.

    Returns ``(backbone, record)`` where ``record['devices']`` logs the
    backbone device the runner callable saw on each invocation.
    """
    _runner._reset_cpu_fallback_latch()
    backbone = _FakeBackbone()
    record: dict = {"devices": [], "calls": 0}

    monkeypatch.setattr(_runner, "get_adapter", lambda name: object())
    monkeypatch.setattr(_runner, "_ensure_runner_imported", lambda name: None)
    monkeypatch.setattr(
        _runner, "load_shared_backbone", lambda *a, **k: backbone
    )
    monkeypatch.setattr(_runner, "LoRAAdapter", _FakeAdapter)

    def _oom_then_ok(*, adapter, inputs, **kwargs):
        record["calls"] += 1
        dev = adapter.backbone.device
        record["devices"].append(dev)
        if dev != "cpu":
            # CUDA-OOM-shaped error while still on the GPU.
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 512.00 MiB"
            )
        return _real_output()

    monkeypatch.setitem(_runner.BERT_RUNNERS, "oom_bert", _oom_then_ok)
    yield backbone, record
    _runner._reset_cpu_fallback_latch()


def test_forward_retried_on_cpu_and_returns_real_result(_wire):
    backbone, record = _wire

    out = _runner.run_bert("oom_bert", ["span"])

    # (a) the forward was retried on CPU: first attempt saw cuda (OOM), the
    #     retry saw cpu.
    assert record["devices"] == ["cuda", "cpu"]
    assert backbone.force_to_cpu_calls >= 1
    # (b) a REAL, non-empty, non-skipped result was returned (not None).
    assert out is not None
    assert isinstance(out, BertOutput)
    assert len(out.signals) == 1
    # (c) the latch is set after the OOM.
    assert _runner.cpu_fallback_latched() is True


def test_second_forward_goes_straight_to_cpu(_wire):
    backbone, record = _wire

    # First forward OOMs on cuda then retries on cpu (2 callable invocations).
    _runner.run_bert("oom_bert", ["span"])
    assert record["calls"] == 2
    assert _runner.cpu_fallback_latched() is True

    # Second forward: latched → straight to CPU, no CUDA re-attempt/re-OOM.
    record["devices"].clear()
    prior_force_calls = backbone.force_to_cpu_calls
    out2 = _runner.run_bert("oom_bert", ["span"])

    assert record["devices"] == ["cpu"]  # exactly one attempt, on CPU
    assert isinstance(out2, BertOutput)
    # force_to_cpu is (idempotently) re-asserted at load while latched.
    assert backbone.force_to_cpu_calls > prior_force_calls


def test_non_oom_exception_propagates_without_latch(_wire, monkeypatch):
    backbone, record = _wire

    def _boom(*, adapter, inputs, **kwargs):
        record["calls"] += 1
        raise ValueError("unrelated failure")

    monkeypatch.setitem(_runner.BERT_RUNNERS, "boom_bert", _boom)

    with pytest.raises(ValueError, match="unrelated failure"):
        _runner.run_bert("boom_bert", ["span"])

    # A non-OOM error is NOT retried and does NOT latch the council to CPU.
    assert record["calls"] == 1
    assert backbone.force_to_cpu_calls == 0
    assert _runner.cpu_fallback_latched() is False


# ---------------------------------------------------------------------------
# The OOM detector itself.
# ---------------------------------------------------------------------------


def test_is_cuda_oom_matches_dedicated_class():
    exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert _runner.is_cuda_oom(exc) is True


@pytest.mark.parametrize(
    "msg",
    [
        "CUDA out of memory. Tried to allocate ...",
        "CUDA error: out of memory",
        "RuntimeError: CUDA out of memory",
    ],
)
def test_is_cuda_oom_matches_string_shapes(msg):
    # A generic RuntimeError / AcceleratorError-shaped OOM (some torch
    # versions don't raise the dedicated class).
    assert _runner.is_cuda_oom(RuntimeError(msg)) is True


def test_is_cuda_oom_rejects_host_memory_error():
    # A host allocation failure (CPU-only box) has an empty str and must NOT
    # be misclassified as a retry-on-CPU CUDA OOM.
    assert _runner.is_cuda_oom(MemoryError()) is False
    assert _runner.is_cuda_oom(ValueError("bad shape")) is False
