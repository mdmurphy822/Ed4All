"""Math helpers for embedding similarity.

``cosine_similarity`` is the np.ndarray-aware port of the stdlib helper
at ``Trainforge/eval/key_term_precision.py:74-80``. The original works
on plain ``Sequence[float]`` inputs; the embedding-tier callers operate
on numpy vectors that the SentenceTransformer model emits, so this
helper accepts ``np.ndarray`` directly and short-circuits zero-norm
vectors to ``0.0`` (the same edge-case behavior as the stdlib version).

The helper is import-safe without the embedding extras installed —
``numpy`` is a transitive dep of pyshacl which is already in the dev
extras, so the import never fails on a baseline checkout. The narrow
typed signature keeps mypy happy without a runtime numpy dependency at
import time of `lib.embedding`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  — type-only import


def cosine_similarity(a: "Any", b: "Any") -> float:
    """Cosine similarity of two equal-length vectors.

    Accepts ``np.ndarray`` or any sequence-of-floats. Zero-norm
    vectors return ``0.0`` to mirror the precedent at
    ``Trainforge/eval/key_term_precision.py:74-80``.

    The runtime path uses numpy when available (the canonical case
    when the embedding extras are installed); falls back to a pure-
    Python loop otherwise so the helper stays importable on a slim
    install.
    """
    try:
        import numpy as _np  # local import keeps `lib.embedding` slim

        a_arr = _np.asarray(a, dtype=_np.float64)
        b_arr = _np.asarray(b, dtype=_np.float64)
        na = float(_np.linalg.norm(a_arr))
        nb = float(_np.linalg.norm(b_arr))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(a_arr, b_arr) / (na * nb))
    except ImportError:
        # Stdlib fallback — exact port of key_term_precision.py:74-80.
        import math

        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a))
        nb = math.sqrt(sum(float(y) * float(y) for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
