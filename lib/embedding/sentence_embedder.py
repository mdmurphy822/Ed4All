"""Lazy-loaded sentence embedder wrapper.

Wraps ``sentence-transformers`` behind a thin abstraction so callers
don't import the heavy ML stack directly. ``try_load_embedder()``
returns ``None`` when the extras are missing — downstream validators
fall back to a warning-severity GateIssue (``EMBEDDING_DEPS_MISSING``)
per Phase 4 Subtask 8. Strict-mode opt-in via
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` flips the policy to critical:
``try_load_embedder()`` raises :class:`EmbeddingDepsMissing` instead
of swallowing the import error. Mirrors the precedent in
``lib/validators/shacl_runner.py:557-576`` (graceful degrade by
default, opt-in fail-closed).

Default model: ``all-MiniLM-L6-v2`` (384-dim, ~80 MB on disk, the same
model the precedent at ``Trainforge/eval/key_term_precision.py:66-71``
uses for stdlib-fallback embedding similarity).

Public surface:
- :class:`EmbeddingDepsMissing` — raised by strict-mode callers.
- :class:`SentenceEmbedder` — model wrapper with ``encode(text)``.
- :class:`EmbeddingCache` — content-addressed LRU on disk (Subtask 6).
- :func:`try_load_embedder` — returns a ``SentenceEmbedder`` or ``None``.
- :func:`is_strict_mode` — reads the ``TRAINFORGE_REQUIRE_EMBEDDINGS`` flag.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np  # noqa: F401

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_CACHE_PATH = Path("runtime/state/embedding_cache.jsonl")
#: Directory the model-folded persistent batch cache (:meth:`encode_batch_cached`)
#: writes under. A NEW filename family (``embedding_cache.batch.<slug>.jsonl``,
#: distinct from ``_DEFAULT_CACHE_PATH``) because the batch cache folds the model
#: name into its key — reusing the model-agnostic ``encode()`` cache file would
#: silently cross model vocabularies.
_DEFAULT_BATCH_CACHE_DIR = Path("state")
_MAX_CACHE_ENTRIES = 100_000

# ------------------------------------------------------------------------- #
# Builder-C incremental-embedding opt-in flags. All default OFF → the encode
# paths are byte-identical to the legacy calls (parse-with-fallback: only the
# explicit truthy tokens enable a flag). Rows live in the root-owned
# docs/operations/behavior-flags.md ED4ALL_* table.
#
# * ED4ALL_EMBED_BATCH_TUNE — when truthy the feature-cache + groundedness _S1
#   embed callers pass ``batch_size=256, length_sort=True`` into encode_batch /
#   encode_batch_cached (larger GPU batches + longest-first packing to cut
#   padding waste). Off → today's exact ``encode_batch(texts)`` calls.
# * ED4ALL_EMBED_PERSIST_CACHE — when truthy those same callers route through
#   :meth:`SentenceEmbedder.encode_batch_cached` (disk-persisted, content +
#   model addressed) instead of the in-memory-only :meth:`encode_batch`. Off →
#   in-memory only, exactly as today.
# * ED4ALL_EMBED_FP16 — when truthy a freshly-loaded model that landed on CUDA is
#   cast to fp16 (``model.half()``). CPU models are left untouched. Cosine
#   similarities shift only in low-order bits (~1e-3 abs) — well under every
#   verdict/rank threshold that consumes these vectors.
ENV_EMBED_BATCH_TUNE = "ED4ALL_EMBED_BATCH_TUNE"
ENV_EMBED_PERSIST_CACHE = "ED4ALL_EMBED_PERSIST_CACHE"
ENV_EMBED_FP16 = "ED4ALL_EMBED_FP16"

#: Batch size + packing knobs the ``ED4ALL_EMBED_BATCH_TUNE`` callers apply.
_TUNE_BATCH_SIZE = 256

# Subtask 8: strict-mode opt-in flag. When truthy,
# ``try_load_embedder()`` raises :class:`EmbeddingDepsMissing` on
# missing extras instead of returning ``None``. Truthy values match the
# convention used by the existing ED4ALL / TRAINFORGE env-var family
# (``true`` / ``1`` / ``yes`` / ``on``, case-insensitive).
_STRICT_MODE_ENV_VAR = "TRAINFORGE_REQUIRE_EMBEDDINGS"
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})

# ------------------------------------------------------------------------- #
# Process-level SentenceEmbedder singleton cache (NLI-reload fix rec #1).
#
# ``try_load_embedder`` historically constructed a BRAND-NEW SentenceEmbedder
# on every call. Every embedding-tier validator's ``validate()`` calls it
# (rewrite_source_grounding, objective_assessment_similarity,
# concept_example_similarity, objective_roundtrip_similarity,
# distractor_misconception_alignment, ...), so across one
# ``post_rewrite_validation`` / eval phase the ~103-tensor MiniLM reloaded
# several times per process — a genuine in-process reload bug independent of
# the resume-loop process churn. This module-level cache (keyed on
# ``model_name``, guarded by a lock — mirroring
# ``NliClassifier.get_or_load``'s pattern) holds each model loaded exactly
# ONCE per process: the first ``try_load_embedder(model_name)`` constructs the
# wrapper, every subsequent call returns the same instance. The validators
# already share the returned object safely (each only reads;
# ``SentenceTransformer.encode`` is thread-safe for these callers). The actual
# heavy ``SentenceTransformer`` load is still deferred to the first
# ``encode()`` inside the wrapper, so the cache adds no eager cost.
_EMBEDDER_CACHE: Dict[str, "SentenceEmbedder"] = {}
_EMBEDDER_CACHE_LOCK = threading.Lock()


def _reset_embedder_cache_for_tests() -> None:
    """Test-only seam — clear the process-level embedder singleton cache."""
    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


def is_strict_mode() -> bool:
    """Return True when ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is truthy.

    Mirrors `Courseforge/scripts/blocks.py::_EMIT_BLOCKS_TRUTHY` —
    case-insensitive match against the canonical truthy values used
    elsewhere in the codebase.
    """
    raw = os.environ.get(_STRICT_MODE_ENV_VAR, "").strip().lower()
    return raw in _TRUTHY_VALUES


def resolve_embed_batch_tune(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_EMBED_BATCH_TUNE`` is a truthy token (default OFF)."""
    src = env if env is not None else os.environ
    return str(src.get(ENV_EMBED_BATCH_TUNE, "")).strip().lower() in _TRUTHY_VALUES


def resolve_embed_persist_cache(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_EMBED_PERSIST_CACHE`` is a truthy token (default OFF)."""
    src = env if env is not None else os.environ
    return str(src.get(ENV_EMBED_PERSIST_CACHE, "")).strip().lower() in _TRUTHY_VALUES


def resolve_embed_fp16(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_EMBED_FP16`` is a truthy token (default OFF)."""
    src = env if env is not None else os.environ
    return str(src.get(ENV_EMBED_FP16, "")).strip().lower() in _TRUTHY_VALUES


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
        batch_cache: Optional["EmbeddingCache"] = None,
    ) -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._cache = cache
        # Disk-persisted, model-folded batch cache for :meth:`encode_batch_cached`
        # (constructed lazily on first use; injectable for tests).
        self._batch_cache = batch_cache

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            self._maybe_half(self._model)
        return self._model

    @staticmethod
    def _maybe_half(model: Any) -> None:
        """Cast a CUDA-resident model to fp16 when ``ED4ALL_EMBED_FP16`` is on.

        CPU models can't run half reliably, so this gates on a cuda device and is
        a no-op otherwise. Best-effort — any failure leaves fp32 intact, so a real
        encode never fails because of the flag. Cosine similarities shift only in
        low-order bits (~1e-3 abs), well under every downstream threshold.
        """
        if not resolve_embed_fp16():
            return
        try:
            dev = getattr(model, "device", None)
            if dev is not None and getattr(dev, "type", "") == "cuda":
                model.half()
        except Exception as exc:  # noqa: BLE001 — never fail a real load
            logger.debug("ED4ALL_EMBED_FP16 half() skipped: %s", exc)

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
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None,
        length_sort: bool = False,
    ) -> "np.ndarray":
        """Batch-encode a list of texts; mirrors :meth:`encode`.

        Bypasses the in-memory ``encode()`` cache to keep the batch path tight;
        callers that want disk-persisted hits use :meth:`encode_batch_cached`.

        ``batch_size`` (default ``None`` → the library default) is forwarded to
        ``SentenceTransformer.encode``. ``length_sort`` (default ``False``, so
        the legacy call is byte-identical) sorts the texts longest-first before
        encoding and restores the caller's input order on return: longest-first
        packing minimises intra-batch padding when the caller batches large,
        length-heterogeneous pools. Both are opt-in knobs the
        ``ED4ALL_EMBED_BATCH_TUNE`` callers set.
        """
        model = self._ensure_model()
        kwargs: Dict[str, Any] = {"normalize_embeddings": normalize}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if not length_sort or len(texts) < 2:
            return model.encode(texts, **kwargs)

        # Longest-first, then invert the permutation to restore input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        reordered = [texts[i] for i in order]
        encoded = model.encode(reordered, **kwargs)
        restored: List[Any] = [None] * len(texts)
        for pos, orig_i in enumerate(order):
            restored[orig_i] = encoded[pos]
        try:
            import numpy as _np

            return _np.stack(restored)
        except Exception:  # noqa: BLE001 — numpy absent → list of vectors
            return restored  # type: ignore[return-value]

    def encode_batch_cached(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None,
        length_sort: bool = False,
    ) -> List[Any]:
        """Batch-encode ``texts`` through a disk-persisted, model-folded cache.

        Consults the content-addressed :class:`EmbeddingCache` per text (key folds
        the model name so two models never collide), batch-encodes only the misses
        (honouring ``batch_size`` / ``length_sort``), appends the misses to the
        cache JSONL (the class appends incrementally so a crash keeps what was
        computed), and returns the vectors in the caller's input order.

        Only reached when ``ED4ALL_EMBED_PERSIST_CACHE`` is truthy; the default
        path stays on the in-memory-only :meth:`encode_batch`.
        """
        if not texts:
            return []
        cache = self._ensure_batch_cache()
        out: List[Any] = [None] * len(texts)
        miss_texts: List[str] = []
        miss_positions: List[int] = []
        first_seen: Dict[str, int] = {}  # keyed_text → index into miss_texts
        for i, text in enumerate(texts):
            keyed = self._batch_cache_key(text)
            cached = cache.get(keyed)
            if cached is not None:
                out[i] = cached
                continue
            if keyed in first_seen:
                # Duplicate miss within this call — encode once, fan out below.
                out[i] = None
                miss_positions.append(i)
                continue
            first_seen[keyed] = len(miss_texts)
            miss_texts.append(text)
            miss_positions.append(i)
        if miss_texts:
            vectors = self.encode_batch(
                miss_texts,
                normalize=normalize,
                batch_size=batch_size,
                length_sort=length_sort,
            )
            for uniq_idx, text in enumerate(miss_texts):
                cache.put(self._batch_cache_key(text), vectors[uniq_idx])
            for i in miss_positions:
                keyed = self._batch_cache_key(texts[i])
                out[i] = cache.get(keyed)
        return out

    def _ensure_batch_cache(self) -> "EmbeddingCache":
        if self._batch_cache is None:
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name).strip("_") or "model"
            path = _DEFAULT_BATCH_CACHE_DIR / f"embedding_cache.batch.{slug}.jsonl"
            self._batch_cache = EmbeddingCache(cache_path=path)
        return self._batch_cache

    def _batch_cache_key(self, text: str) -> str:
        """Model-folded cache key text (``EmbeddingCache`` hashes it to sha256).

        Folding ``model_name`` in means the batch cache file can never serve a
        vector from a different model even though the on-disk row is just
        ``{"hash", "vector"}``.
        """
        return f"{self.model_name}\x00{text}"


class EmbeddingCache:
    """Content-addressed LRU cache for embedding vectors.

    Keyed on ``sha256(text)``. Persists to a JSONL file (default
    ``runtime/state/embedding_cache.jsonl``) — one row per
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
    embedding deps).

    When :func:`is_strict_mode` is true (via the
    ``TRAINFORGE_REQUIRE_EMBEDDINGS`` env flag), the function raises
    :class:`EmbeddingDepsMissing` instead of returning ``None`` so
    operators who explicitly require embedding-tier validation
    fail-closed on missing extras rather than silently degrade to
    warning-severity GateIssues. Mirrors the strict-mode pattern in
    ``lib/validators/shacl_runner.py:557-576``.

    Process-singleton (NLI-reload fix rec #1): the returned
    :class:`SentenceEmbedder` for a given ``model_name`` is cached at module
    level and reused, so the model loads exactly once per process no matter
    how many embedding-tier validators call this within a phase. Only a
    successfully-probed wrapper is cached — a strict-mode raise or a ``None``
    degrade is never cached (a later call re-probes).
    """
    cached = _EMBEDDER_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        # Probe-import; the actual model load is deferred to encode().
        import sentence_transformers  # type: ignore  # noqa: F401
    except ImportError as exc:
        if is_strict_mode():
            raise EmbeddingDepsMissing(
                f"sentence-transformers is not installed but "
                f"{_STRICT_MODE_ENV_VAR} is set: install via "
                f"`pip install -e .[embedding]`. Underlying error: {exc}"
            ) from exc
        logger.debug(
            "sentence-transformers not installed (%s); "
            "try_load_embedder returning None",
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — match precedent at key_term_precision.py:70
        if is_strict_mode():
            raise EmbeddingDepsMissing(
                f"sentence-transformers import raised an unexpected error "
                f"and {_STRICT_MODE_ENV_VAR} is set: {exc}"
            ) from exc
        logger.warning(
            "sentence-transformers import raised unexpected error: %s; "
            "try_load_embedder returning None",
            exc,
        )
        return None

    with _EMBEDDER_CACHE_LOCK:
        # Re-check inside the lock — another thread may have finished the
        # construction while we probed (mirrors NliClassifier.get_or_load).
        existing = _EMBEDDER_CACHE.get(model_name)
        if existing is not None:
            return existing
        embedder = SentenceEmbedder(model_name=model_name)
        _EMBEDDER_CACHE[model_name] = embedder
        return embedder
