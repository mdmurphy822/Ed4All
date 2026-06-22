"""Central registry mapping council BERT name -> adapter spec.

Phase 0: empty registry. Each per-BERT phase (Phase 2 onward) inserts its
own entry. The dict is module-level rather than YAML-driven because the
mapping is code-shaped (Path objects, label tuples) — config.yaml carries
hyperparameters, registry.py carries the disk pointers and head shapes.
"""

from __future__ import annotations

from .base import LoRAAdapterSpec


# Empty in Phase 0. Populated by:
#   Phase 2  -> "math_detector"
#   Phase 3  -> "role", "span", "order", "math_class",
#               "table_class", "semantic"
#   See architecture.md §3.1 for the full council membership.
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
