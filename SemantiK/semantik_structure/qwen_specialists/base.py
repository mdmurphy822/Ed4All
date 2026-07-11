"""AdapterSwap — serial-only Qwen LoRA adapter context manager.

The 3060 has 8 GB of VRAM. Loading two Qwen LoRA adapter contexts at the
same time poisons the CUDA allocator and the second adapter silently
emits garbage tokens — see feedback_qwen_build_serial.md.

`AdapterSwap` is a process-level mutex over adapter loading. It is NOT
re-entrant: nested entry (whether by the same instance or two distinct
instances) raises immediately rather than blocking, because in this
pipeline a nested swap is always a programming error, never a legitimate
wait.

Phase 0 landed the lock + entry/exit semantics. Stage-6 v1 wires the
swap to a pluggable :class:`QwenRuntime` (see ``runtime.py``):

    * ``__enter__`` resolves the adapter's ``gguf_path`` from
      ``config.yaml`` and calls ``runtime.load(path)`` IF a path is
      configured. v1 ships every adapter at ``adapter_path: null`` —
      that's a deliberate signal that the trained GGUF doesn't exist
      yet, and the swap becomes a structural no-op (mutex held, no
      load) so the runner can still walk the 3-pass shape.

    * ``__exit__`` calls ``runtime.free()`` symmetrically. ``free()``
      is safe to call against a runtime that never loaded.

The :func:`generate` module-level helper is the sole dispatch point
the runner uses; it asserts the active adapter matches the request,
routes through the runtime, and wraps each completion in a typed
:class:`Candidate`.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml

from .. import paths as _semantik_paths
from .types import AdapterID, Candidate, SpecialistRequest


logger = logging.getLogger(__name__)


# Module-level sentinel + lock. Both must be module-level (not class-level)
# so two distinct AdapterSwap("foo") and AdapterSwap("bar") instances see
# the same state. A simple non-reentrant lock works: the test for nested
# entry needs to fail loudly, not deadlock.
_swap_lock = threading.Lock()
_active_adapter: AdapterID | str | None = None
_active_runtime: Any | None = None


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class AdapterSwapError(RuntimeError):
    """Raised when AdapterSwap detects nested entry."""


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read ``config.yaml`` (or the override path) into a dict.

    Returned dict has the top-level keys ``version``, ``base_model``,
    ``adapters``. Callers typically only need ``adapters[adapter_id]``.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _gguf_path_for(adapter: AdapterID | str, cfg: dict[str, Any]) -> Path | None:
    """Return the adapter's configured GGUF path, or ``None`` if the
    config explicitly leaves it null (the v1 default — no trained
    adapter on disk).
    """
    adapter_key = adapter.value if isinstance(adapter, AdapterID) else str(adapter)
    adapters = cfg.get("adapters") or {}
    entry = adapters.get(adapter_key) or {}
    raw_path = entry.get("adapter_path")
    if raw_path is None:
        return None
    # config.yaml stores ``models/<...>.gguf`` (CWD-relative historically).
    # Anchor through the SemantiK model root so the GGUF loads regardless of CWD
    # and honors SEMANTIK_MODEL_DIR; byte-stable to the legacy layout otherwise.
    return _semantik_paths.resolve_model_literal(raw_path)


class AdapterSwap:
    """Context manager enforcing one-Qwen-adapter-at-a-time.

    Usage::

        from .runtime import make_runtime

        runtime = make_runtime("mock")  # or "real" when GGUFs exist
        with AdapterSwap(AdapterID.PROSE, runtime=runtime):
            # adapter loaded; safe to call qwen_specialists.base.generate(...)
            ...
        # adapter released; another AdapterSwap may now enter

    Nested entry (any second ``__enter__`` before the first ``__exit__``)
    raises :class:`AdapterSwapError`. This is intentional — see module
    docstring.
    """

    def __init__(
        self,
        adapter: AdapterID | str,
        *,
        runtime: Any | None = None,
        config_path: Path | None = None,
    ) -> None:
        # Accept either the enum or its string value, for ergonomics.
        if isinstance(adapter, AdapterID):
            self.adapter = adapter
        else:
            try:
                self.adapter = AdapterID(adapter)
            except ValueError:
                # Allow arbitrary string identifiers in tests / Phase 0.
                # Production code must pass an AdapterID.
                self.adapter = adapter  # type: ignore[assignment]
        self._entered = False
        self._runtime = runtime
        self._config_path = config_path
        self._loaded = False

    def __enter__(self) -> "AdapterSwap":
        global _active_adapter, _active_runtime
        # Try-acquire; if the lock is held, we have nested entry.
        acquired = _swap_lock.acquire(blocking=False)
        if not acquired:
            raise AdapterSwapError(
                f"nested AdapterSwap detected: tried to enter {self.adapter!r} "
                f"while {_active_adapter!r} is still active. Serial-only — "
                f"see feedback_qwen_build_serial.md."
            )
        _active_adapter = self.adapter
        _active_runtime = self._runtime
        self._entered = True

        # Load the runtime if one was provided AND a GGUF is configured.
        # In v1 every adapter ships with adapter_path=null — this branch
        # logs the no-op but does NOT crash, so the driver can still
        # walk its 3-pass shape with a mock runtime.
        # If load() raises after we acquired the lock, we MUST release
        # the lock or every subsequent AdapterSwap in this process will
        # die with a nested-entry false positive. Wrap in try/except and
        # roll back on any exception.
        try:
            if self._runtime is not None:
                try:
                    cfg = _load_config(self._config_path)
                    gguf = _gguf_path_for(self.adapter, cfg)
                except FileNotFoundError:
                    gguf = None
                if gguf is None:
                    logger.debug(
                        "AdapterSwap(%s): adapter_path is null in config; "
                        "skipping runtime.load (v1 mock-mode behaviour)",
                        self.adapter,
                    )
                else:
                    self._runtime.load(gguf)
                    self._loaded = True
        except BaseException:
            _active_adapter = None
            _active_runtime = None
            self._entered = False
            self._loaded = False
            _swap_lock.release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        global _active_adapter, _active_runtime
        if not self._entered:
            return
        # Symmetric free: only invoke if we actually loaded. Mock runtimes
        # tolerate free() with no prior load, but real runtimes can call
        # torch.cuda.empty_cache() which is wasteful when nothing happened.
        if self._runtime is not None and self._loaded:
            try:
                self._runtime.free()
            except Exception as exc_release:  # noqa: BLE001
                # Don't let a free() error mask the original exception (if any).
                print(
                    f"[AdapterSwap] runtime.free() raised: {exc_release}",
                    file=sys.stderr,
                )
        _active_adapter = None
        _active_runtime = None
        self._entered = False
        self._loaded = False
        _swap_lock.release()


def active_adapter() -> AdapterID | str | None:
    """Return the currently-loaded adapter, or None if no swap is active.

    Useful for assertions in callers that must run inside a swap.
    """
    return _active_adapter


def active_runtime() -> Any | None:
    """Return the runtime bound to the active swap (or None)."""
    return _active_runtime


def generate(
    request: SpecialistRequest,
    *,
    k: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 512,
    seed: int | None = None,
    repeat_penalty: float = 1.0,
) -> list[Candidate]:
    """Run a SpecialistRequest through the active adapter and return K Candidates.

    Must be called inside an :class:`AdapterSwap` context. The active
    adapter (from the swap) MUST equal ``request.adapter`` — mismatched
    routing is a programming error and raises.

    The runtime (mock or real) attached to the swap actually produces
    the K completions; this function is just the typed wrapper that
    folds raw strings into :class:`Candidate` instances.
    """
    if _active_adapter is None or _active_runtime is None:
        raise RuntimeError(
            "qwen_specialists.base.generate() called outside an AdapterSwap "
            "context (or with no runtime attached)"
        )
    if _active_adapter != request.adapter:
        raise RuntimeError(
            f"AdapterSwap active = {_active_adapter!r}, but request asks for "
            f"{request.adapter!r}. Routing mismatch — fix the runner."
        )

    prompt_text = (request.payload or {}).get("prompt")
    if not prompt_text:
        raise ValueError(
            "SpecialistRequest.payload['prompt'] is empty — prompt builders "
            "must set the formatted prompt string before dispatch"
        )

    raw = _active_runtime.generate(
        prompt_text,
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        repeat_penalty=repeat_penalty,
    )

    # Pull the metadata that the prompt builders stash under
    # sampling_overrides["metadata"]; pass it through to Candidate
    # raw_metadata so downstream gates / rerankers can use it.
    builder_metadata = (request.sampling_overrides or {}).get("metadata") or {}

    out: list[Candidate] = []
    for i, text in enumerate(raw):
        out.append(
            Candidate(
                adapter=request.adapter,
                request_id=request.request_id,
                text=text,
                sampling_seed=(seed + i) if seed is not None else None,
                finish_reason="stop",
                raw_metadata={
                    **builder_metadata,
                    "candidate_idx": i,
                },
            )
        )
    return out


__all__ = [
    "AdapterSwap",
    "AdapterSwapError",
    "active_adapter",
    "active_runtime",
    "generate",
]
