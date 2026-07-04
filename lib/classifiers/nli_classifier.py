"""Singleton-load NLI classifier wrapping
``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``.

GPT Feedback v2 Wave 2 W2.F lands the real loader; the prior W1.7.C
stub returned ``None`` from :meth:`NliClassifier.get_or_load` so
:class:`lib.validators.block_objective_delivery.
BlockObjectiveDeliveryValidator`'s graceful-degrade path was the only
exercised arm. Wave 2 W2.F lands the real DeBERTa-v3-mnli-fever-anli
loader; the surface contract is unchanged so the W1.7.C consumer
keeps working without modification.

Surface contract (frozen for downstream consumers):

* :meth:`NliClassifier.get_or_load` is the singleton accessor — returns
  the process-wide instance, lazy-loaded on first access. When
  ``transformers`` / ``torch`` extras are absent (or the model load
  fails for any reason), returns ``None`` so the validator's
  graceful-degrade path (warning-severity GateIssue with
  ``passed=True, action=None``) fires.
* :meth:`NliClassifier.score_pair` returns one :class:`NliScore` for a
  ``(premise, hypothesis)`` pair.
* :meth:`NliClassifier.score_batch` returns a list of scores for a list
  of pairs (batched forward pass for fan-out across many block /
  objective / claim pairs).

Three downstream consumers fan out to this surface:

1. :class:`lib.validators.claim_support.ClaimSupportValidator`
   (Wave 2 W2.F primary brief — claim ↔ chunks).
2. :class:`lib.validators.block_objective_delivery.
   BlockObjectiveDeliveryValidator` (Wave 1.7 W1.7.C — block ↔
   objective; statement-entailment axis only).
3. ``ObjectiveClaimSupportValidator`` (post-Wave-1.6 sibling —
   objective ↔ chunks; deferred to a Wave 4 follow-up because the
   structural seam is sufficient day-1).

The pinned HuggingFace revision is imported from
:data:`lib.classifiers.bloom_bert_ensemble._DEFAULT_ENSEMBLE_MEMBERS`
(the third entry, the same DeBERTa-v3-mnli-fever-anli model used as
the BERT ensemble's zero-shot member) so the NLI loader and the BERT
ensemble can never drift apart on the underlying model commit SHA.

Graceful degrade is the default behavior: missing extras OR a load
failure of any kind returns ``None`` from :meth:`get_or_load`. There
is no strict-mode flag at the loader layer — strict mode lives at the
validator layer (``TRAINFORGE_REQUIRE_EMBEDDINGS``) so a CPU-only dev
box stays usable for everything except the strict-mode validator
runs.

Performance notes:

* Singleton-load — every consumer that imports this module shares the
  same in-memory model. The ~750 MB DeBERTa-v3-base load happens once
  per process; subsequent ``get_or_load()`` calls are O(1).
* Batched scoring — :meth:`score_batch` runs a single forward pass
  for up to :data:`_DEFAULT_BATCH_SIZE` (8) pairs at a time. Block ×
  objective and per-claim × per-chunk fan-outs emit dozens of pairs
  per ``validate()`` call, so the batch path keeps wall-time bounded.
* Inference uses ``torch.no_grad()`` context — no gradients are
  accumulated since this is pure inference.

Device selection (``ED4ALL_NLI_DEVICE``):

* Default ``"cpu"`` — preserves the historical behavior byte-for-byte
  (determinism + CI hermeticity). The model is loaded fp32 and tensors
  stay CPU-resident; NO ``.to(device)`` / ``.half()`` is invoked.
* ``"cuda"`` / ``"cuda:N"`` — the groundedness/eval NLI scoring runs on
  GPU (~20-50x faster than the CPU path on the ~184M-param DeBERTa-v3
  head). On CUDA the model is cast to fp16 (``.half()``) to keep the
  VRAM footprint small (~0.4 GB) — the card is commonly shared with a
  local ollama server (qwen-7b ~5.3 GB on an 8 GB GPU). fp16 is
  CUDA-only; CPU stays fp32 because fp16 on CPU is slow/unsupported.
* Graceful fallback: if ``cuda`` is requested but
  ``torch.cuda.is_available()`` is False, the loader logs a one-time
  warning and falls back to CPU rather than crashing (mirrors the
  embedding provider's device handling) — important since CI and many
  dev boxes have no GPU.
* VRAM contention (``ED4ALL_NLI_EVICT_FOR_CUDA``, default on): when cuda
  is requested + available but free VRAM is below
  ``ED4ALL_NLI_MIN_FREE_VRAM_MIB`` (a resident local ollama generation
  model is holding the card), the loader FIRST evicts that ollama model
  (via :func:`lib.llm.vram_reclaim.evict_local_llm`) and re-probes free
  VRAM. Generation (ollama 7B) and validation (NLI) don't run at the same
  time within a phase, so the card is handed off — the 7B lazy-reloads on
  its next generation request. NLI then loads on cuda for speed. CPU
  fallback is the LAST RESORT only when eviction is disabled (flag
  falsey → ab0ce44 behavior), fails, or doesn't free enough.
* Determinism note: GPU softmax is non-associative, so the post-softmax
  probabilities can differ from the CPU path by ~1e-6. The downstream
  verdict thresholds (0.70 entailment / 0.50 contradiction) are robust
  to that magnitude, so a CUDA-scored run is NOT a regression versus a
  CPU-scored pin. The resolved device + dtype are recorded on the
  instance (and surfaced via :meth:`device`) so a mixed-provenance
  comparison is detectable.

Mirrors the embedding provider's ``ED4ALL_EMBEDDING_DEVICE`` knob
(``lib/embedding/providers.py``): default ``"cpu"`` for determinism,
``"cuda"`` allowed for speed, recorded for provenance.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


#: Env var selecting the torch device for the NLI model. Default ``"cpu"``
#: (determinism + CI hermeticity, byte-identical to the historical
#: behavior). Accepts ``cpu`` / ``cuda`` / ``cuda:N``. Documented in root
#: docs/operations/behavior-flags.md (indexed in root CLAUDE.md). Mirrors ``ED4ALL_EMBEDDING_DEVICE``.
ENV_DEVICE = "ED4ALL_NLI_DEVICE"

#: Default device — CPU keeps the load fp32 and the tensors CPU-resident
#: with no ``.to()`` / ``.half()`` calls, preserving today's behavior.
_DEFAULT_DEVICE = "cpu"

#: Env var: minimum FREE VRAM (MiB) required on the target CUDA device
#: before the NLI model is allowed onto the GPU. On an 8 GB box shared
#: with a resident local ollama 7B (~5.3 GB), free VRAM can sit at
#: ~200 MiB — enough to *load* the ~0.4 GB fp16 head but NOT enough for a
#: batch-8 × 512-token forward pass, whose activations spike several
#: hundred MiB. That spike raises an UNCAUGHT-at-load ``RuntimeError:
#: CUDA out of memory`` mid-validation, which the orchestrator's broad
#: ``except Exception`` swallows as a tracebackless task failure ("silent
#: death"). This floor moves the decision to load time: when free VRAM is
#: below it, the model loads on CPU (fp32) instead — the heavy generation
#: 7B keeps the GPU, the comparatively light NLI scoring runs on CPU. The
#: 1024 MiB default covers the fp16 weights plus forward-pass activation
#: headroom. Set to ``0`` to disable the floor (force the historical
#: load-time-only OOM guard). Mirrors the ``ED4ALL_NLI_DEVICE`` knob.
ENV_MIN_FREE_VRAM_MIB = "ED4ALL_NLI_MIN_FREE_VRAM_MIB"

#: Default free-VRAM floor (MiB). fp16 DeBERTa-v3-base weights are ~0.4 GB;
#: the remainder is forward-pass activation headroom for the batch-8 ×
#: 512-token scoring path (the OOM trigger observed during
#: post_rewrite_validation on an 8 GB card with a resident ollama 7B).
_DEFAULT_MIN_FREE_VRAM_MIB = 1024

#: Env var: when truthy (default), the VRAM-contention guard first EVICTS
#: the resident local ollama generation model (freeing the card) and
#: re-probes free VRAM before deciding to fall back to CPU. The better
#: design over ab0ce44's pure CPU demotion: generation (ollama 7B) and
#: validation (NLI) do NOT run simultaneously within a phase, so the card
#: can be handed off — the 7B lazy-reloads on its next generation request.
#: Set falsey to skip eviction and use the ab0ce44 pure-CPU-fallback
#: behavior. Documented in docs/operations/behavior-flags.md (indexed in root CLAUDE.md).
ENV_EVICT_FOR_CUDA = "ED4ALL_NLI_EVICT_FOR_CUDA"


def resolve_evict_for_cuda(value: Optional[bool] = None) -> bool:
    """Resolve whether to evict the resident local LLM before CPU fallback.

    Resolution chain: explicit ``value`` arg → ``ED4ALL_NLI_EVICT_FOR_CUDA``
    env → default ``True``. Parse-with-fallback: only the explicit falsey
    tokens (``0`` / ``false`` / ``no`` / ``off``, case-insensitive) disable
    eviction; anything else (including garbage) keeps the default-ON
    coexistence behavior the user wants (cuda NLI).
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(ENV_EVICT_FOR_CUDA)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _cuda_device_index(device: str) -> int:
    """Map a torch device string to a 0-based GPU index for NVML.

    ``"cuda"`` / ``""`` → 0 (the current/default device); ``"cuda:N"`` →
    ``N``. Anything unparseable → 0.
    """
    if ":" in device:
        try:
            return int(device.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            return 0
    return 0


def probe_free_vram_mib_with_source(
    torch_module: Any, device: str
) -> Tuple[Optional[int], str]:
    """Same probe as :func:`probe_free_vram_mib`, plus the source backend.

    Returns ``(free_mib, source)`` where ``source`` is:

    * ``"nvml"`` — the NVML memory read succeeded (the cross-process-correct,
      WSL2-load-bearing number);
    * ``"torch"`` — NVML failed and the value came from the
      ``torch.cuda.mem_get_info`` fallback (correct on native Linux only,
      cross-process-BLIND on WSL2 — it over-reports);
    * ``"unavailable"`` — both backends failed (``free_mib`` is ``None``).

    The branch ORDER mirrors :func:`probe_free_vram_mib` exactly (NVML init →
    handle → memory-info → on ANY NVML failure, ``torch.cuda.mem_get_info`` →
    else None); the returned source tag identifies the branch that actually
    produced the int, so a caller stamping provenance can never mislabel a
    torch-derived number as ``"nvml"``. Best-effort: never raises.
    """
    idx = _cuda_device_index(device)
    # 1) NVML — true global free (correct on WSL2 AND native Linux).
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(mem.free) // (1024 * 1024), "nvml"
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 — NVML missing / no permission / odd build
        pass
    # 2) torch.cuda.mem_get_info — correct on native Linux (NOT WSL2, where
    #    it ignores other processes' allocations).
    try:
        dev_arg = device if device != "cuda" else None
        free_bytes, _total = torch_module.cuda.mem_get_info(dev_arg)
        return int(free_bytes) // (1024 * 1024), "torch"
    except Exception:  # noqa: BLE001
        return None, "unavailable"


def probe_free_vram_mib(torch_module: Any, device: str) -> Optional[int]:
    """Return the GLOBALLY-free VRAM (MiB) on ``device``, or None on failure.

    Prefers NVML (``pynvml`` / ``nvidia-ml-py``) because it reports the
    GPU's true cross-process free memory — the number that matters when a
    SEPARATE process (a local ollama LLM server) holds the card. This is
    load-bearing on WSL2, where ``torch.cuda.mem_get_info()`` does NOT
    account for other processes' allocations (it reports ~5.4 GB "free"
    on an 8 GB card that NVML/nvidia-smi correctly show as ~150 MiB free
    with ollama resident) — so a ``mem_get_info``-only probe would wave
    the NLI model onto a full GPU and OOM anyway. Falls back to
    ``torch.cuda.mem_get_info`` (correct on native Linux) when NVML is
    unavailable, then to None (probe failed → caller proceeds with the
    load-time OOM guard only). Best-effort: never raises.

    Thin delegate over :func:`probe_free_vram_mib_with_source` (the single
    probe implementation) so the value-only and value+source surfaces can
    never drift; behavior is byte-identical to the historical loader (same
    return values, never raises, no logging).
    """
    return probe_free_vram_mib_with_source(torch_module, device)[0]


def resolve_min_free_vram_mib(value: Optional[int] = None) -> int:
    """Resolve the free-VRAM floor (MiB) for the NLI cuda gate.

    Resolution chain (mirrors :func:`resolve_nli_device`): explicit
    ``value`` arg → ``ED4ALL_NLI_MIN_FREE_VRAM_MIB`` env → default
    :data:`_DEFAULT_MIN_FREE_VRAM_MIB`. Parse-with-fallback: a negative /
    non-integer / garbage value falls back to the default; ``0`` is a
    valid value that disables the floor.
    """
    if value is not None:
        try:
            iv = int(value)
            return iv if iv >= 0 else _DEFAULT_MIN_FREE_VRAM_MIB
        except (TypeError, ValueError):
            return _DEFAULT_MIN_FREE_VRAM_MIB
    raw = os.environ.get(ENV_MIN_FREE_VRAM_MIB)
    if raw is None or not raw.strip():
        return _DEFAULT_MIN_FREE_VRAM_MIB
    try:
        iv = int(raw.strip())
        return iv if iv >= 0 else _DEFAULT_MIN_FREE_VRAM_MIB
    except ValueError:
        return _DEFAULT_MIN_FREE_VRAM_MIB


def is_cuda_jit_error(exc: BaseException) -> bool:
    """True iff ``exc`` is a CUDA JIT / nvrtc / PTX *compilation* failure.

    The failure mode this guards against: torch built against one CUDA
    runtime (e.g. ``2.11.0+cu130``) on a box whose JIT path can't discover
    the matching ``libnvrtc-builtins.so.*``, so the DeBERTa forward pass on
    CUDA raises a ``RuntimeError`` naming ``nvrtc`` / ``libnvrtc`` / ``PTX``
    (a JIT/PTX compile error) rather than a memory error. That is a
    device-capability problem the CPU path sidesteps entirely, so the
    classifier degrades to CPU on it.

    Deliberately SPECIFIC so it never swallows the two errors handled
    elsewhere:

    * a genuine CUDA **out-of-memory** (``CUDA out of memory``) — owned by
      the load-time VRAM/eviction guards (``ED4ALL_NLI_MIN_FREE_VRAM_MIB`` /
      ``ED4ALL_NLI_EVICT_FOR_CUDA``); OOM is checked FIRST and returns False;
    * any unrelated ``RuntimeError`` / ``ValueError`` — re-raised untouched.
    """
    msg = str(exc).lower()
    # OOM is NOT a JIT error — the load-time VRAM guards own it. Check first
    # so a message that somehow mentions both never routes to the JIT path.
    if "out of memory" in msg:
        return False
    # nvrtc / libnvrtc / PTX name the JIT runtime-compiler explicitly.
    if "nvrtc" in msg or "libnvrtc" in msg or "ptx" in msg:
        return True
    # A generic CUDA kernel *compilation* failure (JIT compile error) that
    # doesn't spell out nvrtc but pairs "cuda" with a compile verb.
    if "cuda" in msg and "compil" in msg:
        return True
    return False


def _disable_gpu_jit_fuser(torch_module: Any) -> None:
    """Disable torch's GPU TensorExpr (NNC) JIT pointwise fuser so the
    DeBERTa CUDA forward pass never triggers nvrtc runtime-compilation.

    WSL2 failure this prevents: torch's TensorExpr fuser JIT-compiles a
    fused pointwise kernel for DeBERTa's relative-position / log-bucket path
    (the dumped kernel contains ``aten::where`` / ``aten::log``) through
    nvrtc, but on this box nvrtc cannot open its matching
    ``libnvrtc-builtins.so.13.0`` — the ``nvidia/cu13/lib`` wheel dir ships
    the file yet ``libnvrtc.so.13`` carries no RUNPATH and that dir is not
    on the loader path, so the very first CUDA forward pass raises
    ``RuntimeError: nvrtc: error: failed to open
    libnvrtc-builtins.so.13.0``. Turning the fuser off makes torch execute
    the forward pass with EAGER (non-fused) CUDA kernels — no JIT, no nvrtc
    — which is numerically identical (kernel fusion is a pure performance
    optimization, not a correctness change). Applied once at cuda placement
    so the JIT is never attempted; the mid-inference degrade-to-CPU latch
    (:meth:`NliClassifier._downgrade_to_cpu`) stays as the last resort.

    Best-effort + idempotent: ``_jit_set_texpr_fuser_enabled`` /
    ``_jit_override_can_fuse_on_gpu`` are private, version-varying
    ``torch._C`` APIs, so each call is individually guarded against
    ``AttributeError`` (and any other error).
    """
    for setter, arg in (
        ("_jit_set_texpr_fuser_enabled", False),
        ("_jit_override_can_fuse_on_gpu", False),
    ):
        try:
            getattr(torch_module._C, setter)(arg)
        except Exception:  # noqa: BLE001 — private torch API, version-dependent
            pass


def resolve_nli_device(device: Optional[str] = None) -> str:
    """Resolve the NLI torch device string.

    Resolution chain (mirrors ``ED4ALL_EMBEDDING_DEVICE``): explicit
    ``device`` arg → ``ED4ALL_NLI_DEVICE`` env → default ``"cpu"``.

    Returns a normalized device string (``"cpu"`` / ``"cuda"`` /
    ``"cuda:N"``). The graceful CUDA-unavailable fallback is applied at
    load time (where ``torch`` is in hand), not here — this resolver only
    reads config and never imports ``torch``.
    """
    resolved = (
        device
        or os.environ.get(ENV_DEVICE)
        or _DEFAULT_DEVICE
    )
    return resolved.strip() or _DEFAULT_DEVICE


#: Default batch size for :meth:`NliClassifier.score_batch`. Sized to
#: keep memory bounded on CPU-only inference; consumers that need
#: bigger batches can override via the ``batch_size`` kwarg.
_DEFAULT_BATCH_SIZE: int = 8

#: Maximum tokenized sequence length. DeBERTa-v3-base is trained on
#: 512-token sequences; longer (premise, hypothesis) pairs get
#: truncated.
_MAX_SEQUENCE_LENGTH: int = 512


class NliScore:
    """Three-way NLI score (post-softmax probabilities).

    The DeBERTa-v3-mnli-fever-anli head emits logits over the
    ``(entailment, neutral, contradiction)`` triple; this container
    exposes the post-softmax probabilities so consumers can apply
    per-axis thresholds without coupling to the underlying transformer's
    logits API.

    Wave 2 W2.F validators threshold at:

    * ``entailment >= 0.7`` — claim is "entailed" by the cited chunks.
    * ``contradiction >= 0.5`` — claim is "contradicted" by the cited
      chunks (a stronger signal than mere "unsupported").

    The Wave 1.7 W1.7.C consumer (BlockObjectiveDeliveryValidator)
    thresholds entailment at a per-block-type table (default 0.40);
    same NliScore shape, different consumer-side thresholds.
    """

    def __init__(
        self,
        *,
        entailment: float,
        neutral: float,
        contradiction: float,
    ) -> None:
        self.entailment = float(entailment)
        self.neutral = float(neutral)
        self.contradiction = float(contradiction)

    def __repr__(self) -> str:
        return (
            f"NliScore(entailment={self.entailment:.4f}, "
            f"neutral={self.neutral:.4f}, "
            f"contradiction={self.contradiction:.4f})"
        )


class NliClassifier:
    """Process-singleton NLI classifier.

    Wraps ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (the same
    DeBERTa-v3-base-mnli-fever-anli model used as the third member of
    the BERT ensemble in :mod:`lib.classifiers.bloom_bert_ensemble`).

    Singleton-load via :meth:`get_or_load`; the in-memory model is
    shared across every validator that calls into this surface so the
    ~750 MB load cost is paid once per process.
    """

    # Module-level singleton state. The ``_INSTANCE`` field caches the
    # loaded classifier; ``_LOAD_FAILED`` caches the negative result so
    # subsequent ``get_or_load()`` calls don't re-attempt the (slow)
    # import probe on a CPU-only dev box. The lock prevents two threads
    # from racing the model load.
    _INSTANCE: Optional["NliClassifier"] = None
    _LOAD_FAILED: bool = False
    _LOAD_LOCK: threading.Lock = threading.Lock()

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        revision: str,
        id2label: dict,
        device: Optional[str] = None,
    ) -> None:
        """Direct constructor — prefer :meth:`get_or_load` from consumer code.

        ``device`` follows the :func:`resolve_nli_device` chain (explicit
        arg > ``ED4ALL_NLI_DEVICE`` > ``"cpu"``). When the resolved device
        is non-CPU and CUDA is actually available, the model is moved to
        the device and cast to fp16 (``.half()``) to keep the VRAM
        footprint small. When CUDA is requested but unavailable, the
        constructor logs a one-time warning and falls back to CPU (no
        crash). On the default CPU path NO ``.to()`` / ``.half()`` is
        called, so behavior is byte-identical to the historical loader.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._revision = revision
        # Resolve + apply the device. ``_device`` / ``_dtype`` are recorded
        # for provenance (mirrors the embedding manifest recording device).
        self._device, self._dtype = self._place_model_on_device(
            model, torch_module, resolve_nli_device(device)
        )
        # Sticky mid-inference CUDA→CPU downgrade latch (see
        # :meth:`_downgrade_to_cpu`). Set when a forward pass raises a CUDA
        # JIT/nvrtc/PTX compile error (a torch↔nvrtc version mismatch); once
        # set, every subsequent batch scores on CPU without re-attempting
        # cuda, so the classifier never thrashes cuda→cpu per batch.
        self._cuda_jit_downgraded = False
        # Build a label -> index map so we can read the three scores
        # back from the model output regardless of the canonical
        # (entailment / neutral / contradiction) ordering. The
        # MoritzLaurer DeBERTa-v3-mnli-fever-anli model card documents
        # ``{0: "entailment", 1: "neutral", 2: "contradiction"}`` but
        # we resolve it from the model's own id2label config to defend
        # against silent revision-change drift.
        normalized = {str(v).strip().lower(): int(k) for k, v in id2label.items()}
        self._idx_entailment = normalized.get("entailment")
        self._idx_neutral = normalized.get("neutral")
        self._idx_contradiction = normalized.get("contradiction")
        if (
            self._idx_entailment is None
            or self._idx_neutral is None
            or self._idx_contradiction is None
        ):
            raise RuntimeError(
                f"NliClassifier could not resolve (entailment, neutral, "
                f"contradiction) from id2label={id2label!r}; "
                f"normalized={normalized!r}. The pinned revision "
                f"{revision!r} may have shifted its label set."
            )

    @staticmethod
    def _place_model_on_device(
        model: Any,
        torch_module: Any,
        device: str,
    ) -> Tuple[str, str]:
        """Move + (on CUDA) fp16-cast the model. Return (device, dtype).

        * ``device == "cpu"`` — no-op. NO ``.to()`` / ``.half()`` is
          called and the model stays fp32 CPU-resident, byte-identical to
          the historical loader. Returns ``("cpu", "float32")``.
        * ``device`` is a CUDA device AND ``torch.cuda.is_available()`` —
          ``model.to(device).half()`` (fp16 keeps VRAM ~0.4 GB on a card
          shared with a local LLM server). Returns ``(device, "float16")``.
        * CUDA requested but unavailable — log a one-time warning and fall
          back to CPU (no crash). Returns ``("cpu", "float32")``.

        Defensive: any unexpected error during placement is logged and the
        model is left on CPU rather than crashing a run that requested
        CUDA on a box without a working GPU.
        """
        if device == "cpu":
            return "cpu", "float32"

        # Non-CPU device requested (cuda / cuda:N). Guard on availability.
        try:
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 — torch probe is best-effort
            logger.warning(
                "NliClassifier could not probe torch.cuda.is_available() "
                "(%s); falling back to CPU (fp32).",
                exc,
            )
            return "cpu", "float32"

        if not cuda_available:
            logger.warning(
                "NliClassifier device %r requested but torch.cuda is not "
                "available; falling back to CPU (fp32). Set "
                "%s=cpu to silence this, or provision a CUDA device.",
                device, ENV_DEVICE,
            )
            return "cpu", "float32"

        # VRAM-contention guard: on an 8 GB card shared with a resident
        # local ollama 7B (~5.3 GB), free VRAM can sit at ~200 MiB — enough
        # to LOAD the ~0.4 GB fp16 head but NOT enough for a batch-8 × 512
        # forward pass, whose activations spike several hundred MiB and
        # raise an uncaught ``RuntimeError: CUDA out of memory`` mid-scoring
        # (the orchestrator swallows it tracebackless → "silent death").
        # Decide at LOAD time: if free VRAM is below the floor, score on CPU
        # so the generation 7B keeps the GPU. ``floor == 0`` disables this
        # (preserves the historical load-time-only OOM guard below).
        floor_mib = resolve_min_free_vram_mib()
        if floor_mib > 0:
            # NVML-first global free-VRAM probe (load-bearing on WSL2, where
            # torch.cuda.mem_get_info ignores other processes' allocations).
            free_mib = probe_free_vram_mib(torch_module, device)
            if free_mib is None:
                logger.warning(
                    "NliClassifier could not probe free VRAM on %r; "
                    "proceeding with the load-time OOM guard only.",
                    device,
                )
            if free_mib is not None and free_mib < floor_mib:
                # The better design (the user directive): generation
                # (ollama 7B) and validation (NLI) do NOT run at the same
                # time within a phase, so instead of demoting NLI to CPU,
                # EVICT the resident ollama generation model to free the
                # card and run NLI on cuda for speed. The 7B lazy-reloads
                # on its next generation HTTP request (ollama auto-loads on
                # demand), so no explicit reload is needed. CPU fallback is
                # the LAST RESORT only when eviction is disabled, fails, or
                # doesn't free enough. The reclaim knowledge lives in
                # ``lib.llm.vram_reclaim`` so the NLI path stays free of
                # ollama-specific wire details.
                if resolve_evict_for_cuda():
                    try:
                        from lib.llm.vram_reclaim import evict_local_llm

                        evicted = evict_local_llm()
                    except Exception as exc:  # noqa: BLE001 — fully graceful
                        logger.debug(
                            "NliClassifier: VRAM-reclaim helper raised "
                            "(%s); treating as eviction failure.", exc,
                        )
                        evicted = False
                    if evicted:
                        # Re-probe after the hand-off (NVML-first again).
                        reprobed = probe_free_vram_mib(torch_module, device)
                        if reprobed is not None and reprobed >= floor_mib:
                            logger.info(
                                "NliClassifier: evicted resident ollama model "
                                "→ free VRAM %d→%d MiB on %r (floor %d MiB) → "
                                "loading NLI on cuda (fp16). The 7B lazy-"
                                "reloads on its next generation request.",
                                free_mib, reprobed, device, floor_mib,
                            )
                            # Fall through to the .to(device).half() block.
                        else:
                            logger.warning(
                                "NliClassifier: evicted ollama but free VRAM "
                                "is %s MiB on %r (floor %d MiB) — insufficient; "
                                "falling back to CPU (fp32) for NLI scoring.",
                                reprobed, device, floor_mib,
                            )
                            return "cpu", "float32"
                    else:
                        logger.warning(
                            "NliClassifier: eviction failed/no-op (only %d MiB "
                            "free on %r, floor %d MiB, %s); falling back to CPU "
                            "(fp32) for NLI scoring to avoid a forward-pass "
                            "CUDA OOM. Lower %s to override.",
                            free_mib, device, floor_mib, ENV_MIN_FREE_VRAM_MIB,
                            ENV_MIN_FREE_VRAM_MIB,
                        )
                        return "cpu", "float32"
                else:
                    # Eviction disabled (ab0ce44 pure-CPU-fallback behavior).
                    logger.warning(
                        "NliClassifier: only %d MiB free on %r (floor %d MiB, "
                        "%s); a resident local LLM is likely holding the card "
                        "and eviction is disabled (%s). Falling back to CPU "
                        "(fp32) for NLI scoring to avoid a forward-pass CUDA "
                        "OOM. The generation model keeps the GPU. Lower %s to "
                        "override.",
                        free_mib, device, floor_mib, ENV_MIN_FREE_VRAM_MIB,
                        ENV_EVICT_FOR_CUDA, ENV_MIN_FREE_VRAM_MIB,
                    )
                    return "cpu", "float32"

        # Proactively disable torch's GPU TensorExpr JIT fuser BEFORE the
        # first CUDA forward pass so nvrtc runtime-compilation is NEVER
        # attempted. On this WSL2 box torch's fuser JIT-compiles a fused
        # DeBERTa pointwise kernel through nvrtc, which cannot open its
        # matching libnvrtc-builtins.so.13.0 (RUNPATH/loader-path gap) and
        # raises "nvrtc: error: failed to open libnvrtc-builtins.so.13.0" on
        # the first cuda forward — historically forcing the whole process to
        # degrade to CPU. Disabling the fuser makes torch run eager (non-
        # fused) CUDA kernels instead — numerically identical — so NLI stays
        # on cuda. The mid-inference degrade-to-CPU latch remains as the
        # last-resort fallback if any residual JIT path still fails.
        _disable_gpu_jit_fuser(torch_module)
        try:
            # fp16 on CUDA keeps the ~184M-param head at ~0.4 GB VRAM so it
            # coexists with a local ollama LLM on an 8 GB card. fp16 is
            # CUDA-only — never applied on CPU (slow/unsupported there).
            model.to(device)
            model.half()
        except Exception as exc:  # noqa: BLE001 — OOM, driver error, etc.
            logger.warning(
                "NliClassifier failed to move model to %r / cast fp16 "
                "(%s); falling back to CPU (fp32).",
                device, exc,
            )
            return "cpu", "float32"

        logger.info(
            "NliClassifier scoring on %s (fp16) — GPU-accelerated NLI. "
            "Note: GPU softmax is non-associative; probabilities may "
            "differ ~1e-6 from a CPU pin (verdict thresholds are robust).",
            device,
        )
        return device, "float16"

    @property
    def device(self) -> str:
        """Resolved torch device the model scores on (``"cpu"`` / ``"cuda*"``).

        Recorded for provenance so a CUDA-scored run is distinguishable
        from a CPU-scored pin (GPU softmax is non-associative — see the
        module docstring). Mirrors how the embedding manifest records the
        embedding device.
        """
        return self._device

    @property
    def dtype(self) -> str:
        """Resolved model dtype (``"float32"`` on CPU, ``"float16"`` on CUDA)."""
        return self._dtype

    @property
    def cuda_jit_downgraded(self) -> bool:
        """True iff a CUDA JIT/nvrtc error downgraded this instance to CPU.

        Recorded for provenance: a run that started on cuda but degraded to
        CPU mid-inference has ``device == "cpu"`` AND this flag set, so a
        mixed-provenance comparison stays detectable even after the device
        string has flipped back to ``"cpu"``.
        """
        return self._cuda_jit_downgraded

    def _downgrade_to_cpu(self, exc: BaseException) -> None:
        """Move the model to CPU (fp32) after a CUDA JIT/nvrtc forward-pass
        failure and latch the downgrade so it sticks for the process.

        Called once, from :meth:`score_batch`, when a forward pass on a CUDA
        device raises a JIT/nvrtc/PTX compile error (:func:`is_cuda_jit_error`).
        Casts fp16→fp32 because fp16 on CPU is slow/unsupported. Idempotent:
        a second call is a no-op (the flag is already set), so repeated
        batches never re-log or re-move.
        """
        if self._cuda_jit_downgraded:
            return
        logger.warning(
            "NliClassifier: CUDA forward pass raised a JIT/nvrtc/PTX compile "
            "error (%s) — likely a torch↔nvrtc runtime mismatch. Degrading to "
            "CPU (fp32) for the remainder of this process and retrying the "
            "batch. Set %s=cpu to select CPU up front and silence this.",
            exc, ENV_DEVICE,
        )
        try:
            self._model.to("cpu")
            # fp16→fp32: the model was .half()'d on cuda; fp16 on CPU is
            # slow/unsupported, so restore fp32 for the CPU path.
            self._model.float()
        except Exception as move_exc:  # noqa: BLE001 — best-effort; still flip
            logger.warning(
                "NliClassifier: error moving model to CPU during JIT "
                "downgrade (%s); proceeding — the forward pass retry will "
                "surface any residual failure.",
                move_exc,
            )
        # Disable torch's TensorExpr (NNC) JIT fuser. The nvrtc/PTX failure
        # comes from the fuser JIT-compiling a fused pointwise kernel through
        # the mismatched CUDA runtime-compiler; that JIT path is attempted
        # even on the CPU retry (the fuser is a global torch setting), so the
        # move-to-CPU alone does NOT clear it. Turning the fuser off sidesteps
        # JIT compilation entirely — correctness is unchanged (fusion is a
        # pure performance optimization). Best-effort: these are private torch
        # APIs that may differ across versions, so every call is guarded.
        self._disable_jit_fuser()
        self._device = "cpu"
        self._dtype = "float32"
        self._cuda_jit_downgraded = True

    def _disable_jit_fuser(self) -> None:
        """Best-effort disable of torch's JIT pointwise fusers.

        Turning off the TensorExpr fuser (and the on-CPU/on-GPU fuse
        overrides) stops torch from JIT-compiling fused kernels via the
        broken nvrtc runtime-compiler on the CPU retry. Every call is
        individually guarded because these are private, version-varying
        ``torch._C`` APIs.
        """
        # Disable the TensorExpr + GPU-fuse overrides via the shared module
        # helper (same setters the proactive cuda-placement path uses), then
        # additionally disable the on-CPU fuser for the CPU retry path.
        _disable_gpu_jit_fuser(self._torch)
        try:
            self._torch._C._jit_override_can_fuse_on_cpu(False)
        except Exception:  # noqa: BLE001 — private API, version-dependent
            pass

    @classmethod
    def get_or_load(cls) -> Optional["NliClassifier"]:
        """Return the process-singleton instance; lazy-load on first call.

        Returns ``None`` permanently if either:

        * ``transformers`` / ``torch`` extras are not installed, or
        * the model load fails for any reason (network error, missing
          revision, deleted repo, OOM, ...).

        The negative result is cached so subsequent calls are O(1) and
        don't re-attempt the (slow) import probe. Callers MUST handle
        the ``None`` return via the graceful-degrade contract: emit a
        warning-severity GateIssue with ``passed=True, action=None``.

        Mirrors :meth:`lib.classifiers.bloom_bert_ensemble.
        BloomBertEnsemble._load_members`'s probe-then-load contract,
        with the difference that NLI is a single model (not three) so
        the loader returns the classifier instance directly rather than
        a list of members.
        """
        if cls._INSTANCE is not None:
            return cls._INSTANCE
        if cls._LOAD_FAILED:
            return None

        with cls._LOAD_LOCK:
            # Re-check inside the lock — another thread may have
            # finished the load while we were waiting.
            if cls._INSTANCE is not None:
                return cls._INSTANCE
            if cls._LOAD_FAILED:
                return None

            # Pull the pinned revision SHA from the BERT ensemble's
            # registry so the NLI loader and the BERT ensemble can
            # never drift apart on the underlying model commit SHA.
            try:
                from lib.classifiers.bloom_bert_ensemble import (
                    _DEFAULT_ENSEMBLE_MEMBERS,
                )
                deberta_entry = _DEFAULT_ENSEMBLE_MEMBERS[2]
                model_name = deberta_entry["name"]
                revision = deberta_entry.get("revision", "main")
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "NliClassifier could not import the pinned revision "
                    "from bloom_bert_ensemble: %s; aborting load.",
                    exc,
                )
                cls._LOAD_FAILED = True
                return None

            # Probe-import the heavy ML stack. Either missing extras
            # is a graceful-degrade path; both are caught the same way.
            try:
                import torch  # type: ignore
                from transformers import (  # type: ignore
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                logger.debug(
                    "NliClassifier deps missing (%s); get_or_load returning None",
                    exc,
                )
                cls._LOAD_FAILED = True
                return None
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NliClassifier import raised unexpected error: %s; "
                    "get_or_load returning None.",
                    exc,
                )
                cls._LOAD_FAILED = True
                return None

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    revision=revision,
                )
                model.eval()
                # Resolve label ordering from the model's own config so
                # a silent revision shift in id2label is caught at
                # construction time (raises RuntimeError) instead of
                # silently emitting wrong-axis scores.
                id2label = getattr(model.config, "id2label", None) or {}
                instance = cls(
                    model=model,
                    tokenizer=tokenizer,
                    torch_module=torch,
                    revision=revision,
                    id2label=id2label,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NliClassifier model load failed for %s@%s: %s; "
                    "get_or_load returning None.",
                    model_name, revision, exc,
                )
                cls._LOAD_FAILED = True
                return None

            cls._INSTANCE = instance
            return instance

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Test-only seam: clear the singleton state so a test can
        re-exercise the load path with a different mock.

        Production code MUST NOT call this; the singleton is
        process-stable by design.
        """
        cls._INSTANCE = None
        cls._LOAD_FAILED = False

    def score_pair(
        self,
        *,
        premise: str,
        hypothesis: str,
    ) -> NliScore:
        """Score a single ``(premise, hypothesis)`` pair.

        Premise = the source / chunk / block-text we ground against.
        Hypothesis = the candidate claim / objective / statement under
        test. The Wave 2 W2.F primary use is
        ``score_pair(premise=chunk_text, hypothesis=claim_text)``;
        Wave 1.7 W1.7.C uses
        ``score_pair(premise=block_text, hypothesis=objective_statement)``.
        """
        scores = self.score_batch(pairs=[(premise, hypothesis)])
        return scores[0]

    def score_batch(
        self,
        *,
        pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[NliScore]:
        """Batched ``(premise, hypothesis)`` scoring.

        Splits ``pairs`` into chunks of ``batch_size`` (default
        :data:`_DEFAULT_BATCH_SIZE`), runs one forward pass per
        chunk, and softmaxes the per-pair logits into the
        ``(entailment, neutral, contradiction)`` triple.

        Returns a list of :class:`NliScore` of the same length as
        ``pairs``. Empty input is a no-op pass that returns an empty
        list.

        Mid-inference CUDA-fallback safety net: when the forward pass runs
        on a CUDA device and raises a JIT/nvrtc/PTX *compile* error (a
        torch↔nvrtc runtime mismatch — see :func:`is_cuda_jit_error`), the
        model is transparently moved to CPU (fp32) and the batch is retried
        there; the downgrade sticks for the process (:meth:`_downgrade_to_cpu`)
        so subsequent batches don't thrash cuda→cpu. A CUDA out-of-memory
        (owned by the load-time VRAM guards) and any unrelated error are NOT
        swallowed — they propagate unchanged.
        """
        if not pairs:
            return []

        size = int(batch_size) if batch_size and batch_size > 0 else _DEFAULT_BATCH_SIZE
        try:
            return self._run_forward(pairs, size)
        except RuntimeError as exc:
            # Only rescue a CUDA JIT/nvrtc compile failure while still on a
            # GPU device. OOM and unrelated RuntimeErrors fall through and
            # re-raise (``is_cuda_jit_error`` excludes OOM by construction).
            if self._device != "cpu" and is_cuda_jit_error(exc):
                self._downgrade_to_cpu(exc)
                return self._run_forward(pairs, size)
            raise

    def _run_forward(
        self,
        pairs: List[Tuple[str, str]],
        size: int,
    ) -> List[NliScore]:
        """Run the batched forward pass on the current ``self._device``.

        Split out of :meth:`score_batch` so the caller can retry it on CPU
        after a CUDA JIT/nvrtc downgrade without duplicating the tokenize →
        forward → softmax loop. Reads ``self._device`` on every call, so a
        retry after :meth:`_downgrade_to_cpu` transparently runs on CPU.
        """
        results: List[NliScore] = []
        torch = self._torch

        with torch.no_grad():
            for batch_start in range(0, len(pairs), size):
                batch = pairs[batch_start : batch_start + size]
                premises = [str(p) for p, _ in batch]
                hypotheses = [str(h) for _, h in batch]
                # ``return_tensors="pt"`` returns torch tensors; the
                # tokenizer handles padding + truncation automatically.
                encoded = self._tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=_MAX_SEQUENCE_LENGTH,
                    return_tensors="pt",
                )
                # Move the tokenized tensors onto the model's device. On the
                # default CPU path this is a no-op (tensors are already CPU
                # tensors and ``.to("cpu")`` returns the same tensor), so the
                # byte-stable CPU behavior is preserved; on CUDA it ships the
                # batch to the GPU before the forward pass.
                if self._device != "cpu":
                    encoded = {
                        k: v.to(self._device) for k, v in encoded.items()
                    }
                outputs = self._model(**encoded)
                logits = outputs.logits  # shape: (batch, 3)
                # Softmax along the label axis -> per-class probabilities.
                probs = torch.softmax(logits, dim=-1)
                # Pull the probabilities back to CPU before reading floats so
                # ``.item()`` works regardless of where the forward pass ran.
                # No-op on CPU; required on CUDA.
                if self._device != "cpu":
                    probs = probs.cpu()
                # ``probs`` shape: (batch, 3); each row is one pair.
                for i in range(probs.shape[0]):
                    row = probs[i]
                    ent = float(row[self._idx_entailment].item())
                    neu = float(row[self._idx_neutral].item())
                    con = float(row[self._idx_contradiction].item())
                    results.append(
                        NliScore(
                            entailment=ent,
                            neutral=neu,
                            contradiction=con,
                        )
                    )

        return results


__all__ = ["NliClassifier", "NliScore"]
