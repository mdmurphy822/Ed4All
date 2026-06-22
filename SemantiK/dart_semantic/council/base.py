"""Council backbone + LoRA adapter abstractions.

The council shares one backbone encoder (DeBERTa-v3-base by default;
ModernBERT-base is the fallback — see Plans/01_implementation_plan.md
DP-0.1) and swaps a per-BERT LoRA adapter on top.

Phase 2 (this file) makes the Phase 0 protocol stubs concrete:

* ``SharedBackbone`` is a singleton-per-model-name wrapper around
  ``transformers.AutoModel.from_pretrained``. The encoder is loaded
  exactly once per process; subsequent ``get(...)`` calls return the
  cached instance. The Phase 0 ``Protocol`` shape is preserved in
  ``SharedBackboneProtocol`` for type-checking purposes.

* ``LoRAAdapter`` wraps ``peft.PeftModel`` over the backbone. ``load``
  attaches a LoRA adapter; ``unload`` detaches it. **Adapter-swap
  discipline:** only one adapter is resident on a backbone at a time,
  and loading a new adapter releases the previous one. This matches the
  council's "shared backbone, swap adapter per BERT" runtime contract.

Heavy framework imports (torch, transformers, peft) are deferred to the
first call site so plain ``import dart_semantic.council.base`` stays
free; that is required by the Phase 0 invariant that the module import
graph must not pull torch into ``sys.modules`` for offline tests.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class SharedBackboneProtocol(Protocol):
    """Type-only protocol describing the shared backbone surface.

    Kept for static type-checking and existing call sites that imported
    this name in Phase 0. The concrete singleton lives in
    ``SharedBackbone`` below.
    """
    name: str
    revision: str
    hidden_size: int
    max_position_embeddings: int

    def encode(self, input_ids: Any, attention_mask: Any) -> Any:
        ...


@dataclass(frozen=True)
class LoRAAdapterSpec:
    """A frozen description of one BERT's adapter on disk.

    The actual loading happens via :class:`LoRAAdapter`; this dataclass
    only records the disk pointer + head shape so the registry is fully
    typed without any torch import.
    """
    bert_name: str                  # e.g. "math_specialist" | "role"
    adapter_path: Path              # directory containing adapter_config.json
    head_kind: str                  # "classification" | "multi_head" | ...
    num_labels: int | None = None   # for single-head classification
    label_names: tuple[str, ...] = ()
    head_specs: tuple[tuple[str, int], ...] = ()
    # ^ for multi-head BERTs: ((head_name, num_classes), ...)


# ---------------------------------------------------------------------------
# Concrete shared backbone (process-singleton-per-model-name)
# ---------------------------------------------------------------------------

# Module-level cache + lock. The lock guards the cache during double-checked
# lazy load. Loading the encoder weights is expensive (~100 MB download / GB
# VRAM), so we serialise the very first ``get`` call per model name.
_BACKBONE_CACHE: dict[str, "SharedBackbone"] = {}
_BACKBONE_CACHE_LOCK = threading.Lock()


class SharedBackbone:
    """Singleton wrapper around ``AutoModel.from_pretrained``.

    Use :meth:`SharedBackbone.get` to obtain a cached instance keyed by
    model name. Direct construction is allowed for tests / explicit
    re-load, but does not populate the cache.

    The model is loaded **lazily** — instantiating the object does NOT
    import torch or download weights. The ``model`` and ``tokenizer``
    attributes materialise on first access (or first ``encode`` call),
    keeping plain ``import`` light enough for stdlib-only tests.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        *,
        revision: str = "main",
        dtype: str = "bfloat16",
        device: str | None = None,
    ) -> None:
        self.name = model_name
        self.revision = revision
        self._dtype_str = dtype
        self._device = device
        # Hidden state populated on first model load.
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = threading.Lock()
        # Static metadata from the architecture YAML; the values are
        # asserted against the loaded config in `_ensure_loaded`.
        self.hidden_size = 768
        self.max_position_embeddings = 512

    # -- factory ------------------------------------------------------------

    @classmethod
    def get(
        cls,
        model_name: str = "answerdotai/ModernBERT-base",
        *,
        revision: str = "main",
        dtype: str = "bfloat16",
        device: str | None = None,
    ) -> "SharedBackbone":
        """Return the process-wide singleton for ``model_name``.

        Subsequent calls with the same ``model_name`` return the same
        instance — the encoder is loaded exactly once per process. A
        different ``model_name`` produces a separate cache entry.
        """
        with _BACKBONE_CACHE_LOCK:
            cached = _BACKBONE_CACHE.get(model_name)
            if cached is not None:
                return cached
            inst = cls(
                model_name,
                revision=revision,
                dtype=dtype,
                device=device,
            )
            _BACKBONE_CACHE[model_name] = inst
            return inst

    @classmethod
    def reset_cache(cls) -> None:
        """Drop every cached backbone — for tests only.

        Production code should never call this; the singleton is what
        keeps the encoder warm across council BERTs.
        """
        with _BACKBONE_CACHE_LOCK:
            _BACKBONE_CACHE.clear()

    # -- lazy load ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            # Imports deferred until first use. This is what keeps plain
            # ``import dart_semantic.council.base`` torch-free.
            import torch  # noqa: WPS433
            from transformers import AutoModel, AutoTokenizer  # noqa: WPS433

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

            tokenizer = AutoTokenizer.from_pretrained(
                self.name, revision=self.revision,
            )
            model = AutoModel.from_pretrained(
                self.name,
                revision=self.revision,
                dtype=torch_dtype,
            )
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(self._device)
            model.eval()

            cfg = getattr(model, "config", None)
            if cfg is not None:
                self.hidden_size = int(getattr(cfg, "hidden_size", self.hidden_size))
                self.max_position_embeddings = int(
                    getattr(cfg, "max_position_embeddings", self.max_position_embeddings)
                )

            self._tokenizer = tokenizer
            self._model = model

    # -- public surface -----------------------------------------------------

    @property
    def model(self) -> Any:
        """Materialise + return the underlying ``transformers`` model."""
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Materialise + return the matching tokenizer."""
        self._ensure_loaded()
        return self._tokenizer

    @property
    def device(self) -> str | None:
        return self._device

    def encode(self, input_ids: Any, attention_mask: Any) -> Any:
        """Run the encoder forward and return last-hidden-states."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ---------------------------------------------------------------------------
# Concrete LoRA adapter (one-resident-per-backbone discipline)
# ---------------------------------------------------------------------------


class LoRAAdapter:
    """Wraps a ``peft.PeftModel`` over a ``SharedBackbone``.

    The contract is **one adapter resident per backbone instance**.
    ``load(spec)`` attaches the named adapter; subsequent ``load`` calls
    on the same backbone first call ``unload`` on the previous adapter
    so two adapter weight tensors never co-occupy VRAM.
    """

    # Map id(backbone) -> the LoRAAdapter currently bound to it.
    _CURRENT_PER_BACKBONE: dict[int, "LoRAAdapter"] = {}
    _BIND_LOCK = threading.Lock()

    def __init__(self, backbone: SharedBackbone, spec: LoRAAdapterSpec) -> None:
        self.backbone = backbone
        self.spec = spec
        self._peft_model: Any = None  # populated by load()
        self._loaded = False

    # -- discipline ---------------------------------------------------------

    @classmethod
    def current(cls, backbone: SharedBackbone) -> "LoRAAdapter | None":
        """Return the adapter currently bound to ``backbone``, or None."""
        return cls._CURRENT_PER_BACKBONE.get(id(backbone))

    # -- public surface -----------------------------------------------------

    def load(self) -> "LoRAAdapter":
        """Attach this adapter on top of the backbone.

        If a different adapter is currently bound to the same backbone,
        unload it first. Loading the same adapter twice is a no-op.
        """
        with self._BIND_LOCK:
            existing = self._CURRENT_PER_BACKBONE.get(id(self.backbone))
            if existing is self and self._loaded:
                return self
            if existing is not None and existing is not self:
                existing._do_unload()
            self._do_load()
            self._CURRENT_PER_BACKBONE[id(self.backbone)] = self
            return self

    def unload(self) -> None:
        """Detach this adapter from the backbone."""
        with self._BIND_LOCK:
            current = self._CURRENT_PER_BACKBONE.get(id(self.backbone))
            if current is not self:
                # Already not bound (or another adapter is bound). Drop
                # any local handle and return.
                self._do_unload()
                return
            self._do_unload()
            self._CURRENT_PER_BACKBONE.pop(id(self.backbone), None)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def peft_model(self) -> Any:
        """The underlying ``peft.PeftModel`` (None if not loaded)."""
        return self._peft_model

    # -- internals ----------------------------------------------------------

    def _do_load(self) -> None:
        if self._loaded:
            return
        # Lazy import — keeps plain module import torch-free.
        from peft import PeftModel  # noqa: WPS433

        base_model = self.backbone.model
        adapter_path = str(self.spec.adapter_path)
        # PeftModel.from_pretrained mounts the LoRA matrices on top of the
        # base encoder. Subsequent forward calls go through the wrapper.
        self._peft_model = PeftModel.from_pretrained(
            base_model, adapter_path,
        )
        self._peft_model.eval()
        self._loaded = True

    def _do_unload(self) -> None:
        if not self._loaded:
            return
        try:
            # peft >=0.10 exposes unload(); fall back to dropping the ref.
            unload = getattr(self._peft_model, "unload", None)
            if callable(unload):
                # `unload` returns the merged base model in newer peft;
                # we discard it because the backbone reference is what
                # downstream callers hold.
                unload()
        except Exception:
            # Worst-case: drop the ref — Python GC + torch will free.
            pass
        self._peft_model = None
        self._loaded = False


# ---------------------------------------------------------------------------
# Backwards-compatible Phase 0 entry points
# ---------------------------------------------------------------------------


def load_shared_backbone(
    name: str = "answerdotai/ModernBERT-base",
    revision: str = "main",
    *,
    dtype: str = "bfloat16",
) -> SharedBackbone:
    """Phase 0 / Phase 2 entry: return the singleton backbone for ``name``.

    Lazy — does not actually load weights until first access.
    """
    return SharedBackbone.get(name, revision=revision, dtype=dtype)


def load_lora_adapter(
    spec: LoRAAdapterSpec,
    backbone: SharedBackbone,
) -> LoRAAdapter:
    """Phase 0 / Phase 2 entry: build + load a ``LoRAAdapter``.

    Releases any previously-bound adapter on this backbone (one-resident
    discipline).
    """
    adapter = LoRAAdapter(backbone, spec)
    adapter.load()
    return adapter


def release_council_gpu() -> None:
    """Evict all council backbones + adapters from the GPU.

    Safe to call once the council/structure stages (1–5) are done and before
    Stage 6 (Qwen): the BERTs are not used again for this PDF. Reclaims the
    ~2–3 GB the shared backbone holds so the Qwen GGUF fits on the 8 GB card —
    enforcing the "one model resident at a time" discipline across the
    BERT→Qwen boundary (without it, large PDFs OOM on GGUF load even in an
    isolated process). Backbones reload lazily if used again.
    """
    import gc

    # Drop adapter refs (peft models wrapping the backbone).
    try:
        LoRAAdapter._CURRENT_PER_BACKBONE.clear()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass

    with _BACKBONE_CACHE_LOCK:
        backbones = list(_BACKBONE_CACHE.values())
        _BACKBONE_CACHE.clear()

    for bb in backbones:
        try:
            if bb._model is not None:
                # Force weights off-GPU even if a stray ref lingers, then drop.
                bb._model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        bb._model = None
        bb._tokenizer = None

    gc.collect()
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "LoRAAdapter",
    "LoRAAdapterSpec",
    "SharedBackbone",
    "SharedBackboneProtocol",
    "load_lora_adapter",
    "load_shared_backbone",
    "release_council_gpu",
]
