"""Lazy-loaded sentence embedder wrapper.

Wraps ``sentence-transformers`` behind a thin abstraction so callers
don't import the heavy ML stack directly. ``try_load_embedder()``
returns ``None`` when the extras are missing — downstream validators
fall back to a warning-severity GateIssue per Phase 4 Subtask 8.

Default model: ``all-MiniLM-L6-v2`` (384-dim, ~80 MB on disk, the same
model the precedent at ``Trainforge/eval/key_term_precision.py:66-71``
uses for stdlib-fallback embedding similarity).

Public surface:
- :class:`EmbeddingDepsMissing` — raised by strict-mode callers.
- :class:`SentenceEmbedder` — model wrapper with ``encode(text)``.
- :class:`EmbeddingCache` — content-addressed LRU on disk (Subtask 6).
- :func:`try_load_embedder` — returns a ``SentenceEmbedder`` or ``None``.

Subtask 8 will add strict-mode opt-in via the
``TRAINFORGE_REQUIRE_EMBEDDINGS`` env flag.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import numpy as np  # noqa: F401

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_CACHE_PATH = Path("state/embedding_cache.jsonl")
_MAX_CACHE_ENTRIES = 100_000


class EmbeddingDepsMissing(RuntimeError):
    """Raised in strict mode when ``sentence-transformers`` is unavailable.

    Strict mode is opt-in via ``TRAINFORGE_REQUIRE_EMBEDDINGS=true``
    (Subtask 8). Default mode swallows the error and returns ``None``
    from :func:`try_load_embedder` so downstream validators degrade
    to a warning-severity GateIssue instead of failing closed.
    """


class SentenceEmbedder:
    """Wrapper around a SentenceTransformer model.

    Lazy-instantiates the underlying model on first call to
    :meth:`encode`. Constructor never imports the heavy ML stack —
    that's what makes ``try_load_embedder()`` cheap on a slim install.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        cache: Optional["EmbeddingCache"] = None,
    ) -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._cache = cache

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, text: str, normalize: bool = True) -> "np.ndarray":
        """Encode ``text`` to a unit-length embedding vector.

        ``normalize=True`` returns a unit vector so cosine similarity
        reduces to a dot product downstream. ``normalize=False`` is
        rarely useful; it's exposed for tests that want to inspect raw
        magnitudes.

        When a :class:`EmbeddingCache` is wired in, hits short-circuit
        the model call. Misses run through the model and append to the
        cache. ``normalize=False`` skips the cache entirely (raw
        magnitudes are caller-specific and rarely cacheable).
        """
        if self._cache is not None and normalize:
            cached = self._cache.get(text)
            if cached is not None:
                return cached
        model = self._ensure_model()
        # SentenceTransformer.encode accepts ``normalize_embeddings`` directly.
        vector = model.encode(text, normalize_embeddings=normalize)
        if self._cache is not None and normalize:
            self._cache.put(text, vector)
        return vector

    def encode_batch(
        self, texts: List[str], normalize: bool = True
    ) -> "np.ndarray":
        """Batch-encode a list of texts; mirrors :meth:`encode`.

        Bypasses the cache to keep the batch path tight; callers that
        want cache-hits-on-batch should iterate :meth:`encode`.
        """
        model = self._ensure_model()
        return model.encode(texts, normalize_embeddings=normalize)


class EmbeddingCache:
    """Content-addressed LRU cache for embedding vectors.

    Keyed on ``sha256(text)``. Persists to a JSONL file (default
    ``state/embedding_cache.jsonl``) — one row per
    ``{"hash": <hex>, "vector": [<floats>]}``. The cache is loaded
    once per run on construction; misses append to the JSONL as they
    happen so a crash mid-run preserves what was computed before it.

    LRU bound is :data:`_MAX_CACHE_ENTRIES` (default 100 000). When the
    bound is hit, the oldest entry is evicted from the in-memory
    ``OrderedDict``; the JSONL keeps growing on disk and the next
    construction trims to the last ``_MAX_CACHE_ENTRIES`` rows. This
    makes the cache file effectively append-only — no rewrite-in-place
    semantics are needed.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        max_entries: int = _MAX_CACHE_ENTRIES,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, Any]" = OrderedDict()
        self._load_from_disk()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_from_disk(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                rows: List[dict] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "embedding_cache.jsonl row malformed (skipped): %s",
                            exc,
                        )
        except OSError as exc:
            logger.warning("Failed to read embedding cache at %s: %s", self.cache_path, exc)
            return

        # Trim to the last max_entries rows so the in-memory dict
        # respects the LRU bound regardless of on-disk file size.
        if len(rows) > self.max_entries:
            rows = rows[-self.max_entries :]

        for row in rows:
            h = row.get("hash")
            v = row.get("vector")
            if isinstance(h, str) and isinstance(v, list):
                self._entries[h] = self._coerce_vector(v)

    @staticmethod
    def _coerce_vector(v: List[float]) -> Any:
        """Coerce a JSON-decoded list to numpy when available, else list."""
        try:
            import numpy as _np

            return _np.asarray(v, dtype=_np.float32)
        except ImportError:
            return v

    def get(self, text: str) -> Optional[Any]:
        """Return the cached vector for ``text`` or ``None`` on miss."""
        h = self._hash_text(text)
        if h in self._entries:
            # Mark as most-recently-used per LRU semantics.
            self._entries.move_to_end(h)
            return self._entries[h]
        return None

    def put(self, text: str, vector: Any) -> None:
        """Insert ``(text, vector)`` and append to the JSONL on disk."""
        h = self._hash_text(text)
        if h in self._entries:
            self._entries.move_to_end(h)
            return
        # Evict oldest if at bound.
        while len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
        self._entries[h] = vector
        self._append_to_disk(h, vector)

    def _append_to_disk(self, h: str, vector: Any) -> None:
        # Serialize vector to plain list[float] for portability.
        try:
            import numpy as _np

            if isinstance(vector, _np.ndarray):
                vec_list = vector.astype(float).tolist()
            else:
                vec_list = [float(x) for x in vector]
        except ImportError:
            vec_list = [float(x) for x in vector]

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"hash": h, "vector": vec_list}) + "\n")
        except OSError as exc:
            logger.warning(
                "Failed to append to embedding cache at %s: %s",
                self.cache_path,
                exc,
            )

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, text: str) -> bool:
        return self._hash_text(text) in self._entries


def try_load_embedder(
    model_name: str = _DEFAULT_MODEL_NAME,
) -> Optional[SentenceEmbedder]:
    """Return a :class:`SentenceEmbedder` or ``None`` when extras missing.

    Mirrors ``Trainforge/eval/key_term_precision.py:66-71`` (the
    in-tree precedent for graceful-degradation behavior on optional
    embedding deps). Subtask 8 will layer strict-mode raise behavior
    on top of this base function.
    """
    try:
        # Probe-import; the actual model load is deferred to encode().
        import sentence_transformers  # type: ignore  # noqa: F401
    except ImportError as exc:
        logger.debug(
            "sentence-transformers not installed (%s); "
            "try_load_embedder returning None",
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — match precedent at key_term_precision.py:70
        logger.warning(
            "sentence-transformers import raised unexpected error: %s; "
            "try_load_embedder returning None",
            exc,
        )
        return None

    return SentenceEmbedder(model_name=model_name)
