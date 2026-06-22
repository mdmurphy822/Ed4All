"""Common runner: load backbone, swap LoRA, run a multi-head BERT.

``run_bert(name, inputs)`` is the single entry point the council
scheduler uses to invoke any registered BERT. It:

    1. Resolves the registry entry for ``name`` (raises if missing).
    2. Acquires the singleton :class:`SharedBackbone`.
    3. Loads the named LoRA adapter on top — releasing whichever
       adapter was previously bound to the backbone.
    4. Calls into the BERT-specific runner (e.g.
       :mod:`dart_semantic.council.math_specialist`) for tokenisation,
       multi-head forward, and packaging.
    5. Wraps the per-head logits as :class:`TypedSignal`s and returns a
       :class:`BertOutput`.

Each council BERT lives in its own module under ``dart_semantic.council``
and registers a ``run_inputs(adapter, inputs) -> BertOutput`` callable
via :data:`BERT_RUNNERS`. The runner is intentionally tiny and import-
light — it threads the registered callable, never embeds per-BERT logic.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import LoRAAdapter, load_shared_backbone
from .registry import get_adapter
from .types import BertOutput


# Per-BERT runners. Populated by each BERT module on import (see
# ``dart_semantic.council.math_specialist``). The signature is:
#
#     fn(adapter: LoRAAdapter, inputs: Any, *, multihead) -> BertOutput
#
# where ``multihead`` is the ``MultiHeadModel`` instance the runner
# constructs for that BERT.
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

    backbone = load_shared_backbone(
        backbone_name,
        revision=backbone_revision,
        dtype=backbone_dtype,
    )
    adapter = LoRAAdapter(backbone, spec)
    adapter.load()  # releases any previously-bound adapter on the backbone
    fn = BERT_RUNNERS[name]
    return fn(adapter=adapter, inputs=inputs, **runner_kwargs)


__all__ = [
    "BERT_RUNNERS",
    "register_runner",
    "run_bert",
]
