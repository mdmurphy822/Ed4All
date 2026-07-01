"""Shared CUDA out-of-memory detection + free-VRAM probe helpers.

Factored out of ``MCP/core/executor.py`` (W2.1) so BOTH reliability paths
recognise a CUDA OOM the same way and emit a loud, attributable diagnostic
instead of burying it:

* the task-execution path (``MCP/core/executor.py::_execute_with_retries``)
  used these to LOG a loud GPU-OOM line before deferring to the poison-pill
  circuit breaker, and
* the validation-gate path (``MCP/hardening/validation_gates.py``) previously
  swallowed an OOM raised inside a validator via its broad ``except`` — under
  ``behavior_on_error=warn`` that became a SILENT auto-pass. It now reuses
  :func:`is_cuda_oom` to detect the OOM and emit a DISTINCT ``VALIDATOR_OOM``
  gate issue.

A dedicated module is required because ``executor`` imports
``validation_gates`` — importing the helpers the other way would create a
circular import. ``lib.llm`` is the neutral shared home (it already owns the
VRAM-reclaim / VRAM-doctor helpers).

Kept dependency-light: ``torch`` is imported lazily inside the functions,
never at module import, so this module is safe on a CPU / CI box with no
torch. Neither function ever raises.
"""

from __future__ import annotations

from typing import Optional


def is_cuda_oom(exc: Optional[BaseException]) -> bool:
    """Return True iff ``exc`` is (or looks like) a CUDA out-of-memory error.

    A ``torch.cuda.OutOfMemoryError`` raised during an NLI / embedding
    forward pass on a VRAM-starved box (a resident local 7B holding the
    card) is otherwise swallowed by a broad ``except Exception`` and logged
    as a generic, hard-to-grep warning (or, in a validation gate, silently
    auto-passed). This predicate lets callers recognise the OOM and emit a
    LOUD, attributable diagnostic.

    Deliberately does NOT require torch to be importable (torch may be
    absent in CI / on a CPU box). Detection is three-pronged:

      1. ``isinstance`` against ``torch.cuda.OutOfMemoryError`` when torch
         imports cleanly (the precise check).
      2. The exception class name is ``OutOfMemoryError`` (covers a torch
         OOM seen through a different import path, and any builtin
         ``OutOfMemoryError``).
      3. The message contains "cuda out of memory", or BOTH "out of
         memory" AND "cuda" (robust against driver / runtime variants).

    Never raises.
    """
    if exc is None:
        return False
    # 1) Precise isinstance when torch is importable.
    try:  # pragma: no cover - torch presence is environment-dependent
        import torch  # type: ignore

        oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    except Exception:  # noqa: BLE001 - torch absent / broken build
        pass
    # 2) Class-name fallback (torch absent / different import path).
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    # 3) Message-content fallback.
    msg = str(exc).lower()
    if "cuda out of memory" in msg:
        return True
    if "out of memory" in msg and "cuda" in msg:
        return True
    return False


def probe_free_vram_mib() -> Optional[int]:
    """Best-effort free-VRAM (MiB) snapshot for an OOM diagnostic.

    Cheap and GPU-process-free: reads the NLI classifier's
    ``probe_free_vram_mib`` (NVML-first, so it reports the correct
    cross-process free VRAM on WSL2) DIRECTLY. It deliberately does NOT go
    through ``lib/llm/vram_doctor.snapshot_vram`` — that helper does a
    blocking ollama ``/api/ps`` HTTP round-trip plus a total-VRAM probe to
    build a full snapshot, which is far too heavy for a diagnostic that
    only needs one free-MiB int. ``torch`` and the probe are imported
    lazily inside the guard so neither is a hard dependency (torch may be
    absent in CI / on a CPU box). Returns ``None`` on any failure; never
    raises (the diagnostic path must not itself blow up).
    """
    try:
        import torch  # type: ignore

        from lib.classifiers.nli_classifier import probe_free_vram_mib as _probe

        free = _probe(torch, "cuda")
        if free is not None:
            return int(free)
    except Exception:  # noqa: BLE001 - torch / probe unavailable
        pass
    return None
