"""Torch-FREE unit tests — council batch-size resolution + slicing logic.

These cover the pure-Python part of the council batching change (env
resolution + the ``range(0, n, bs)`` partition logic). They do NOT import
torch, so they run on a torch-less dev box; the tensor-equivalence tests that
exercise ``batched_cls_pooled`` end-to-end live in
``test_council_batched_forward.py`` behind an ``importorskip``.
"""

from __future__ import annotations

import math

import pytest

from dart_semantic.council.structure import resolve_council_batch_size


def test_resolve_batch_size_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_COUNCIL_BATCH_SIZE", raising=False)
    assert resolve_council_batch_size() == 32


@pytest.mark.parametrize("bad", ["0", "-3", "garbage", "", "  ", "1.5"])
def test_resolve_batch_size_garbage_fallback(monkeypatch, bad):
    monkeypatch.setenv("SEMANTIK_COUNCIL_BATCH_SIZE", bad)
    assert resolve_council_batch_size() == 32


@pytest.mark.parametrize("val,expected", [("1", 1), ("8", 8), ("16", 16), ("64", 64)])
def test_resolve_batch_size_valid(monkeypatch, val, expected):
    monkeypatch.setenv("SEMANTIK_COUNCIL_BATCH_SIZE", val)
    assert resolve_council_batch_size() == expected


@pytest.mark.parametrize(
    "n,bs",
    [(0, 4), (1, 4), (37, 8), (20, 1), (20, 1000), (12, 5), (5, 5), (6, 5)],
)
def test_slicing_partitions_input_contiguously(n, bs):
    """``range(0, n, bs)`` partitions input in-order, no overlap, no gaps, with
    ceil(n/bs) batches — the batched-output equivalence precondition."""
    texts = list(range(n))
    rebuilt: list[int] = []
    n_batches = 0
    for start in range(0, len(texts), bs):
        chunk = texts[start : start + bs]
        assert len(chunk) <= bs
        rebuilt.extend(chunk)
        n_batches += 1
    assert rebuilt == texts  # order preserved, every element exactly once
    assert n_batches == (math.ceil(n / bs) if n else 0)
