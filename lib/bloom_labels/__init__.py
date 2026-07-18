"""``lib.bloom_labels`` — deterministic Bloom-label harvester (no LLM).

Walks a Courseforge project export (and, optionally, a LibV2 course dir) and
harvests every artifact-asserted Bloom label — the generator's OWN declared
``bloom_level`` on synthesized objectives, outline/rewrite blocks, and
assessment items — into a de-duplicated JSONL label store.

The store is the corpus behind the re-founded ``bloom_classifier_disagreement``
signal's voter 1 ("does the generator's own Bloom claim survive independent
checks"): each row pairs a piece of learner-facing text with the level the
local NVIDIA open-model seat asserted for it. Purely inference/read-only —
no model weights are loaded and no text is rewritten.
"""
from __future__ import annotations

from .harvester import (
    DEFAULT_STORE_PATH,
    LICENSE_TAG,
    SCHEMA_VERSION,
    BloomLabel,
    HarvestResult,
    harvest_bloom_labels,
)

__all__ = [
    "harvest_bloom_labels",
    "HarvestResult",
    "BloomLabel",
    "DEFAULT_STORE_PATH",
    "LICENSE_TAG",
    "SCHEMA_VERSION",
]
