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
- :func:`try_load_embedder` — returns a ``SentenceEmbedder`` or ``None``.

Subtask 6 will extend this module with :class:`EmbeddingCache`.
Subtask 8 will add strict-mode opt-in via the
``TRAINFORGE_REQUIRE_EMBEDDINGS`` env flag.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import numpy as np  # noqa: F401

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


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

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None

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
        """
        model = self._ensure_model()
        # SentenceTransformer.encode accepts ``normalize_embeddings`` directly.
        return model.encode(text, normalize_embeddings=normalize)

    def encode_batch(
        self, texts: List[str], normalize: bool = True
    ) -> "np.ndarray":
        """Batch-encode a list of texts; mirrors :meth:`encode`."""
        model = self._ensure_model()
        return model.encode(texts, normalize_embeddings=normalize)


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
