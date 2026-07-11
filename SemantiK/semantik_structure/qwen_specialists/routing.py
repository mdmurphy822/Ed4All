"""Region-kind → AdapterID mapping for Stage 6.

Strict routing per architecture.md §4: one specialist per region, no
fan-out. The K candidate diversity comes from sampling within the
chosen specialist (temperature / top-p), not from fanning to multiple
adapters.

The mapping is deliberately a flat dict — no inheritance, no fallback
chain. If a kind shows up that isn't in the map, that's a bug in
``structure_graph.py`` (a new region kind without a routing entry) and
we want to crash loudly rather than silently downgrade to prose.

Notes on individual mappings
----------------------------

* ``form`` → PROSE in v1. Forms are a known under-developed slice
  (the prose adapter will produce a labelled-input shape, but a
  dedicated form adapter — or a deterministic emitter that walks the
  pikepdf AcroForm widgets directly — is the right answer long term.
  Revisit when we have form-specific training data or a deterministic
  emitter to short-circuit Stage 6 on form regions entirely.

* ``figure`` → PROSE in v1. The prose adapter handles ``<figure>`` /
  ``<figcaption>`` shape; image alt-text inference is explicitly
  out-of-scope for Stage 6 (architecture.md §4: gap-fill in Stage 9
  doesn't do alt-text either in v1). The prose adapter just emits
  the structural wrapper.

* ``definition_list`` → PROSE. Stage 5 v1 emits no ``definition_list``
  regions today; the entry is here so a future detector doesn't
  trip the unknown-kind raise.

* ``metadata_drop`` → PROSE. Stage 9 surfaces these as artifacts /
  ``<address>`` / ``<footer>``; the prose adapter generates the
  inner HTML for the artifact wrapper.
"""
from __future__ import annotations

from .types import AdapterID


REGION_KIND_TO_ADAPTER: dict[str, AdapterID] = {
    "paragraph":       AdapterID.PROSE,
    "heading":         AdapterID.PROSE,
    "list":            AdapterID.PROSE,
    "definition_list": AdapterID.PROSE,
    "code_block":      AdapterID.PROSE,
    "blockquote":      AdapterID.PROSE,
    "figure":          AdapterID.PROSE,
    "form":            AdapterID.PROSE,           # see module docstring — revisit.
    "metadata_drop":   AdapterID.PROSE,
    "table":           AdapterID.TABLE,
    "math":            AdapterID.MATH,
}


def adapter_for(kind: str) -> AdapterID:
    """Return the AdapterID for a Stage-5 ``Region.kind``.

    Raises ``ValueError`` if ``kind`` isn't in the map — this is
    intentionally fail-loud; silent fallback would let new
    structure_graph kinds bypass routing review.
    """
    if kind not in REGION_KIND_TO_ADAPTER:
        raise ValueError(
            f"unknown region kind: {kind!r} — add an entry to "
            f"REGION_KIND_TO_ADAPTER or fix structure_graph.py"
        )
    return REGION_KIND_TO_ADAPTER[kind]


__all__ = [
    "REGION_KIND_TO_ADAPTER",
    "adapter_for",
]
