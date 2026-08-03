"""Lazy adapter registry for the five-specialist compatibility council.

The current specialists are ``merge_or_split``, ``structure``, ``semantic``,
``table_specialist``, and ``math_specialist``. Each specialist registers its
code-shaped adapter specification when imported; ``config.yaml`` carries
shared runtime hyperparameters. Keeping the registry module-level preserves
lazy imports and prevents package import from loading model weights.
"""

from __future__ import annotations

from .base import LoRAAdapterSpec


# Populated lazily by the five specialist modules named in the docstring.
ADAPTER_REGISTRY: dict[str, LoRAAdapterSpec] = {}


def register_adapter(spec: LoRAAdapterSpec) -> None:
    """Add a council BERT to the registry. Idempotent on identical spec."""
    existing = ADAPTER_REGISTRY.get(spec.bert_name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"adapter {spec.bert_name!r} already registered with a different spec; "
            f"existing={existing}, new={spec}"
        )
    ADAPTER_REGISTRY[spec.bert_name] = spec


def get_adapter(bert_name: str) -> LoRAAdapterSpec:
    """Fetch a registered adapter spec or raise KeyError."""
    if bert_name not in ADAPTER_REGISTRY:
        raise KeyError(
            f"no council BERT registered as {bert_name!r}; known names: {sorted(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[bert_name]


def list_adapters() -> list[str]:
    """Return all registered BERT names in deterministic order."""
    return sorted(ADAPTER_REGISTRY)
