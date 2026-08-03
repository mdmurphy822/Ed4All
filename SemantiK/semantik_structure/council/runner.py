"""Common runner: load backbone, swap LoRA, run a multi-head BERT.

``run_bert(name, inputs)`` is the single entry point the council
scheduler uses to invoke any registered BERT. It:

    1. Resolves the registry entry for ``name`` (raises if missing).
    2. Acquires the singleton :class:`SharedBackbone`.
    3. Loads the named LoRA adapter on top — releasing whichever
       adapter was previously bound to the backbone.
    4. Calls into the BERT-specific runner (e.g.
       :mod:`semantik_structure.council.math_specialist`) for tokenisation,
       multi-head forward, and packaging.
    5. Wraps the per-head logits as :class:`TypedSignal`s and returns a
       :class:`BertOutput`.

Each council BERT lives in its own module under ``semantik_structure.council``
and registers a ``run_inputs(adapter, inputs) -> BertOutput`` callable
via :data:`BERT_RUNNERS`. The runner is intentionally tiny and import-
light — it threads the registered callable, never embeds per-BERT logic.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from .base import LoRAAdapter, load_shared_backbone
from .registry import get_adapter
from .types import BertOutput


# ---------------------------------------------------------------------------
# GPU-contention CPU fallback (D14)
# ---------------------------------------------------------------------------
#
# When another process pins the whole GPU (e.g. a vLLM-served LLM holding all
# VRAM), an in-process council backbone forward raises a CUDA out-of-memory
# error deep inside the model. Historically the orchestrator's ``_safe_run``
# caught that exception and SILENTLY SKIPPED the head — the run still reported
# ``gates=pass`` while losing e.g. merge_or_split heading refinement on a
# chapter (a silent-degradation defect). Instead, we detect the CUDA-OOM class
# specifically, move the shared backbone to CPU, and RE-RUN the forward there
# so the refinement is preserved. Once an OOM is hit we LATCH to CPU for the
# rest of the process so subsequent span-batches go straight to CPU rather than
# re-attempting CUDA and re-OOMing on every batch.
_CPU_FALLBACK_LATCH = False

_FORCE_CPU_ENV = "SEMANTIK_COUNCIL_FORCE_CPU"
_FORCE_CPU_LOGGED = False


def resolve_council_force_cpu() -> bool:
    """PRE-EMPTIVE council-on-CPU gate (``SEMANTIK_COUNCIL_FORCE_CPU``).

    Default OFF → byte-identical: the council runs on the GPU and only falls
    back to CPU reactively via the D14 CUDA-OOM latch. Truthy
    (``1``/``true``/``yes``/``on``, case-insensitive) → the council loads on CPU
    from the FIRST head, skipping the OOM-crash-then-recover dance entirely.

    Motivation: when a co-resident model (e.g. a vLLM-served LLM sized at a high
    ``--gpu-memory-utilization``) already pins most of the card, the reactive
    latch can crash PAST the fallback (a hard OOM during the Structure-head load
    surfaces as "no Structure cascade available", a degraded conversion) rather
    than cleanly recovering. Pre-forcing CPU trades council speed for a
    non-degraded structure pass while the GPU stays leased to the VLM/authoring
    seat — the intended posture on a big-memory concurrent-serving host (Spark).
    Read at CALL time (tests/Docker can flip it without a re-import). Parse-with-
    fallback: unset / blank / garbage → off.
    """
    return str(os.environ.get(_FORCE_CPU_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def cpu_fallback_latched() -> bool:
    """True when the council must run on CPU for this process.

    Either a CUDA OOM has reactively latched CPU (D14), OR the operator
    pre-forced it via ``SEMANTIK_COUNCIL_FORCE_CPU`` (checked at call time so a
    contended-GPU run degrades gracefully instead of crashing).
    """
    if _CPU_FALLBACK_LATCH:
        return True
    if resolve_council_force_cpu():
        global _FORCE_CPU_LOGGED
        if not _FORCE_CPU_LOGGED:
            _FORCE_CPU_LOGGED = True
            _warn(
                "council pre-forced to CPU via SEMANTIK_COUNCIL_FORCE_CPU — GPU "
                "left leased to the VLM/authoring seat; council refinement runs "
                "on CPU (slower) but non-degraded."
            )
        return True
    return False


def _reset_cpu_fallback_latch() -> None:
    """Clear the process-scoped CPU-fallback latch. Test-only."""
    global _CPU_FALLBACK_LATCH, _FORCE_CPU_LOGGED
    _CPU_FALLBACK_LATCH = False
    _FORCE_CPU_LOGGED = False


def _warn(msg: str) -> None:
    """Loud, unconditional stderr warning (mirrors orchestrator._log)."""
    print(f"[council] {msg}", file=sys.stderr, flush=True)


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True iff ``exc`` is a CUDA out-of-memory error.

    Matches the specific ``torch.cuda.OutOfMemoryError`` class when torch is
    importable AND defensively string-matches ``out of memory`` /
    ``CUDA error: out of memory`` in the exception text — some torch /
    accelerator versions surface a CUDA OOM as ``torch.AcceleratorError`` or a
    plain ``RuntimeError`` rather than the dedicated class. A host
    ``MemoryError`` (empty ``str``) never matches, so a genuinely CPU-only
    allocation failure is NOT misclassified as a retry-on-CPU OOM.
    """
    try:
        import torch  # noqa: WPS433 — deferred; keeps bare import torch-free

        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:  # noqa: BLE001 — torch absent / import failure
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


# Per-BERT runners. Populated by each BERT module on import (see
# ``semantik_structure.council.math_specialist``). The signature is:
#
#     fn(adapter: LoRAAdapter, inputs: Any, **runner_kwargs) -> BertOutput
#
# Each specialist owns its tokenizer, feature projection, and output heads;
# the generic runner owns backbone and adapter lifecycle only.
BERT_RUNNERS: dict[str, Callable[..., BertOutput]] = {}


def register_runner(bert_name: str, fn: Callable[..., BertOutput]) -> None:
    """Register a BERT-specific runner. Idempotent on identical fn."""
    existing = BERT_RUNNERS.get(bert_name)
    if existing is not None and existing is not fn:
        raise ValueError(
            f"runner for {bert_name!r} already registered as {existing!r}; "
            f"refusing to clobber with {fn!r}"
        )
    BERT_RUNNERS[bert_name] = fn


def _ensure_runner_imported(bert_name: str) -> None:
    """Best-effort import the per-BERT module so it can self-register.

    Avoids forcing every council module to be eagerly imported by
    ``__init__.py``, which would defeat the lazy-load invariant.
    """
    if bert_name in BERT_RUNNERS:
        return
    if bert_name == "math_specialist":
        from . import math_specialist  # noqa: F401, WPS433
    elif bert_name == "merge_or_split":
        from . import merge_or_split  # noqa: F401, WPS433
    elif bert_name == "structure":
        from . import structure  # noqa: F401, WPS433
    elif bert_name == "semantic":
        from . import semantic  # noqa: F401, WPS433
    elif bert_name == "table_specialist":
        from . import table_specialist  # noqa: F401, WPS433
    # Future BERTs are added here as new `elif bert_name == ...` branches.


def run_bert(
    name: str,
    inputs: Any,
    *,
    backbone_name: str = "answerdotai/ModernBERT-base",
    backbone_revision: str = "main",
    backbone_dtype: str = "bfloat16",
    **runner_kwargs: Any,
) -> BertOutput:
    """Run the named council BERT against ``inputs``.

    Loads the shared backbone (singleton), swaps in the LoRA adapter
    registered for ``name``, dispatches to the per-BERT runner.

    Any additional keyword arguments are forwarded verbatim to the
    per-BERT ``run_inputs`` callable. Each per-BERT runner is free to
    accept (or ignore — via ``**kwargs``) these passthroughs. For
    example, the orchestrator forwards ``top_k_per_head=...`` to
    Structure's runner so the cascade-bound heads serialize their full
    softmax distributions.
    """
    # Importing the per-BERT module triggers `register_adapter` AND
    # `register_runner` on import, so this must happen before either
    # the registry or runner lookup.
    _ensure_runner_imported(name)
    spec = get_adapter(name)
    if name not in BERT_RUNNERS:
        raise KeyError(f"no runner registered for {name!r}; known: {sorted(BERT_RUNNERS)}")
    fn = BERT_RUNNERS[name]

    def _load_and_run() -> BertOutput:
        backbone = load_shared_backbone(
            backbone_name,
            revision=backbone_revision,
            dtype=backbone_dtype,
        )
        # If a prior forward already OOM'd, go straight to CPU (no re-attempt
        # on CUDA — that would just re-OOM on every span-batch).
        if cpu_fallback_latched():
            backbone.force_to_cpu()
        adapter = LoRAAdapter(backbone, spec)
        adapter.load()  # releases any previously-bound adapter on the backbone
        return fn(adapter=adapter, inputs=inputs, **runner_kwargs)

    try:
        return _load_and_run()
    except Exception as exc:  # noqa: BLE001 — narrowed below to CUDA OOM only
        # Only a CUDA OOM on a not-yet-latched (i.e. still-on-GPU) forward is
        # retried on CPU. An already-latched forward runs on CPU, so any error
        # there is genuine — and a non-OOM error keeps the historic
        # propagate-to-orchestrator behavior (``_safe_run`` logs + records it).
        if cpu_fallback_latched() or not is_cuda_oom(exc):
            raise
        global _CPU_FALLBACK_LATCH
        _warn(
            f"council BERT {name!r} fell back to CPU after CUDA OOM — GPU is "
            f"pinned by another model; refinement continues on CPU (slower). "
            f"exc={exc!r}"
        )
        _CPU_FALLBACK_LATCH = True
        # force_to_cpu() runs inside _load_and_run() now that the latch is set,
        # moving the backbone off the GPU and rebuilding the adapter on CPU.
        return _load_and_run()


__all__ = [
    "BERT_RUNNERS",
    "cpu_fallback_latched",
    "is_cuda_oom",
    "register_runner",
    "run_bert",
]
