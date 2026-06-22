"""Shared dataset split utility.

Re-exports the canonical ``stable_split_for_id`` from
``scripts/build_semantic_preservation_dataset.py`` so all dataset
builders that key on ``arxiv_id`` produce coherent splits — i.e. an
arxiv_id that lands in val/test for the semantic_preservation
cross-encoder lands in val/test here too. Without that coherence,
training one head on data the other head was held-out on creates
silent leakage when the two heads run together at inference.

We import rather than copy: keeping a single source of truth means a
future change to bucket boundaries propagates everywhere. The
upstream function is pure (sha1 of the id, no side effects) and
imported lazily so this module stays cheap.
"""
from __future__ import annotations

import hashlib


def stable_split_for_id(
    arxiv_id: str, train_frac: float = 0.80, val_frac: float = 0.10
) -> str:
    """Hash-based split. SHA1(arxiv_id) -> [0, 1) -> bucket.

    Identical implementation to
    ``scripts.build_semantic_preservation_dataset.stable_split_for_id``;
    duplicated as a tiny pure function so importing this utility does
    not transitively import the (heavy) semantic-preservation builder.
    A unit-test in ``tests/test_qwen_math_dataset_contract.py`` pins
    the two implementations to the same outputs.
    """
    h = hashlib.sha1(arxiv_id.encode("utf-8")).hexdigest()
    val = int(h[:8], 16) / 0xFFFFFFFF
    if val < train_frac:
        return "train"
    if val < train_frac + val_frac:
        return "val"
    return "test"


__all__ = ["stable_split_for_id"]
