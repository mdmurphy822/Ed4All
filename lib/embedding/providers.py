"""Embedding provider registry — registry entries, NOT subclasses.

Mirrors the OpenAI-compatible LLM-provider precedent at
``MCP/orchestrator/llm_backend.py::_OPENAI_COMPATIBLE_PROVIDERS`` (one
class, registry entries only, env-var resolution chain, ``unverified``
flag). The same operator standing rule applies here: a new embedding
backend is a registry-entry change to ``_EMBEDDING_PROVIDERS``, never a
new subclass.

This registry is DELIBERATELY separate from the unified chat-LLM
endpoint registry (``config/endpoints.yaml`` / ``lib/llm/endpoints.py``):
embeddings live on a different transport axis (``st`` in-process
sentence-transformers vs. ``openai-embeddings`` HTTP), carry
embedding-only metadata (``device`` / ``batch_size`` /
``trust_remote_code`` / per-model license + task-prefix registry), and
fail closed with their own ``EmbeddingBackendUnavailable`` /
anti-poisoning (``ED4ALL_EMBEDDING_ALLOW_FAKE``) semantics — none of
which belong in a chat-completions endpoint row. Folding the two would
conflate two unrelated config axes, so they stay siblings.

Three execution kinds:

- ``"st"`` — in-process ``sentence-transformers`` (the ``[embedding]``
  pyproject extra). Reuses ``lib/embedding/sentence_embedder.py``'s lazy
  model load through a thin wrapper. ``offline=True`` (the query-path
  default) forces ``local_files_only=True`` / ``HF_HUB_OFFLINE=1`` so a
  query can NEVER trigger a network download — absence of cached weights
  raises :class:`EmbeddingBackendUnavailable`, never a silent download.
- ``"openai-embeddings"`` — HTTP ``POST {base_url}/embeddings`` (OpenAI
  wire shape) against a LOCAL server (Ollama ``:11434/v1``, vLLM
  ``:8000/v1``, llama.cpp ``:8080/v1``). A thin httpx client with the
  retry policy mirrored from
  ``Trainforge/generators/_openai_compatible_client.py::DEFAULT_RETRY_STATUS_CODES``;
  the chat client there is NOT reused (it owns ``/chat/completions``
  only) — this module cross-references it as the pattern source.
- ``"fake"`` — deterministic test provider (``sha256(text)`` → seeded
  RNG → L2-normalized unit vector, dim 32). A REAL registry entry so the
  whole build/read/benchmark path is exercisable in CI with zero
  weights and zero network. Production-poisoning is blocked: index
  manifests record the provider and the query path refuses
  ``provider="fake"`` unless ``ED4ALL_EMBEDDING_ALLOW_FAKE`` is set
  (mirrors the ``LOCAL_DISPATCHER_ALLOW_STUB`` precedent).

Honest failure: a missing model / down server raises
:class:`EmbeddingBackendUnavailable`. NO code path may catch it and fall
back to lexical / a different provider / a stub vector — that is the
anti-silent-degradation contract this whole work-stream exists to
enforce.

Public surface (frozen — downstream WS2 executors code against it):

- :class:`EmbeddingBackendUnavailable`
- :class:`ResolvedEmbeddingProvider`
- :func:`resolve_embedding_provider`
- :class:`EmbeddingClient`
- :func:`build_embedding_client`

Benchmark candidates + license metadata: :data:`EMBEDDING_MODEL_REGISTRY`
(all Apache-2.0 / MIT; see ``docs/LICENSING.md`` § "Embedding
providers").
"""
from __future__ import annotations

import hashlib
import logging
import os
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np  # noqa: F401 — type-only

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Truthy-env helper — same convention as
# ``lib/embedding/sentence_embedder.py::_TRUTHY_VALUES``.
# ---------------------------------------------------------------------------
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_VALUES


# ---------------------------------------------------------------------------
# Env-var names (single source of truth for the ED4ALL_EMBEDDING_* family).
# Documented in root CLAUDE.md § "Cross-cutting flags" + gui/env_catalog.py.
# ---------------------------------------------------------------------------
ENV_PROVIDER = "ED4ALL_EMBEDDING_PROVIDER"
ENV_MODEL = "ED4ALL_EMBEDDING_MODEL"
ENV_BASE_URL = "ED4ALL_EMBEDDING_BASE_URL"
ENV_API_KEY = "ED4ALL_EMBEDDING_API_KEY"
ENV_DEVICE = "ED4ALL_EMBEDDING_DEVICE"
ENV_BATCH_SIZE = "ED4ALL_EMBEDDING_BATCH_SIZE"
ENV_ALLOW_FAKE = "ED4ALL_EMBEDDING_ALLOW_FAKE"

DEFAULT_PROVIDER = "st"
_FAKE_DIM = 32


# ---------------------------------------------------------------------------
# Provider registry — registry entries, NOT subclasses (operator standing
# rule). Mirrors ``_OPENAI_COMPATIBLE_PROVIDERS`` field conventions.
# ---------------------------------------------------------------------------
_EMBEDDING_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "st": {
        "kind": "st",
        "model_env": ENV_MODEL,
        "model_default": "BAAI/bge-large-en-v1.5",  # re-pinned from the 2026-06-10 4-model benchmark (hybrid-rrf winner)
        "batch_size_env": ENV_BATCH_SIZE,
        "batch_size_default": 16,
        "device_env": ENV_DEVICE,  # default "cpu" — determinism (D4)
        "device_default": "cpu",
    },
    "local-openai": {
        "kind": "openai-embeddings",
        "base_url_env": ENV_BASE_URL,
        "base_url_default": "http://localhost:11434/v1",
        "api_key_env": ENV_API_KEY,
        "api_key_default": "local",
        "model_env": ENV_MODEL,
        "model_default": "nomic-embed-text",  # Apache-2.0, ollama-pullable
        "batch_size_env": ENV_BATCH_SIZE,
        "batch_size_default": 16,
        "api_key_required": False,
    },
    "fake": {
        "kind": "fake",
        "model_default": "fake-deterministic-v1",
        "batch_size_default": 16,
        "dim": _FAKE_DIM,
    },
}


# ---------------------------------------------------------------------------
# Candidate model registry with license metadata (D5). All Apache-2.0 / MIT.
# ``trust_remote_code`` is recorded for gte-large so it's operator-visible
# and the candidate can be dropped without touching the harness (risk R2).
# ``query_prefix`` / ``document_prefix`` carry the nomic task prefixes (D6).
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BAAI/bge-m3": {
        "license": "MIT",
        "dim": 1024,
        "trust_remote_code": False,
        "notes": "multilingual headroom; former st default (unmeasured — superseded by the benchmarked bge-large pin)",
    },
    "BAAI/bge-large-en-v1.5": {
        "license": "MIT",
        "dim": 1024,
        "trust_remote_code": False,
        # bge retrieval expects the query-side instruction prefix; passages
        # get NO prefix (document_prefix stays empty). Asymmetric, query-only.
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "notes": "current st default — selected by the 2026-06-10 benchmark (hybrid-rrf, positive MRR delta on all 3 corpora); query-side instruction prefix only",
    },
    "BAAI/bge-base-en-v1.5": {
        "license": "MIT",
        "dim": 768,
        "trust_remote_code": False,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "notes": "MIT vanilla en-only; smaller bge sibling, query-side prefix only",
    },
    "Alibaba-NLP/gte-large-en-v1.5": {
        "license": "Apache-2.0",
        "dim": 1024,
        "trust_remote_code": True,  # executes arbitrary HF code — droppable candidate (R2)
        "notes": "requires trust_remote_code=True",
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "license": "Apache-2.0",
        "dim": 768,
        "trust_remote_code": True,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "notes": "also Ollama-servable as nomic-embed-text; needs task prefixes",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "license": "Apache-2.0",
        "dim": 384,
        "trust_remote_code": False,
        "notes": "already cached; CI real-model smoke + floor baseline only",
    },
    # Stretch candidate (D5). 8B oversized for the benchmark loop; 0.6B optional.
    "Qwen/Qwen3-Embedding-0.6B": {
        "license": "Apache-2.0",
        "dim": 1024,
        "trust_remote_code": False,
        "notes": "documented stretch candidate; optional 0.6B variant",
    },
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class EmbeddingBackendUnavailable(RuntimeError):
    """Weights/server absent or refused. NEVER triggers a lexical fallback.

    Raised when:

    - the ``st`` model weights are not cached and the client is offline
      (``offline=True``), so the load would otherwise hit the network;
    - a ``st`` model load fails for any reason;
    - the ``openai-embeddings`` server is unreachable / returns a
      non-retryable error / a malformed body;
    - ``sentence-transformers`` is not installed for a ``st`` provider.

    Callers MUST let this propagate (the query path re-raises it as
    ``EmbeddingBackendUnavailable`` verbatim). No code path may catch it
    and return lexical / stub results.
    """


# ---------------------------------------------------------------------------
# Resolved provider (frozen contract)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedEmbeddingProvider:
    """A fully-resolved embedding provider (post env-var resolution).

    Immutable so it can be hashed / stashed on a client without
    re-resolution drift. ``device`` is ``st``-only; ``base_url`` /
    ``api_key`` are ``openai-embeddings``-only — each is ``None`` for the
    other kinds.
    """

    provider_name: str
    kind: str  # "st" | "openai-embeddings" | "fake"
    model_id: str
    device: Optional[str]
    base_url: Optional[str]
    batch_size: int
    api_key: Optional[str] = None
    trust_remote_code: bool = False
    query_prefix: str = ""
    document_prefix: str = ""


def resolve_embedding_provider(
    provider_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> ResolvedEmbeddingProvider:
    """Resolve a registry entry into a :class:`ResolvedEmbeddingProvider`.

    Resolution chain mirrors ``resolve_openai_compatible_backend``:
    explicit arg → env var → registry default. ``provider_name`` defaults
    to ``ED4ALL_EMBEDDING_PROVIDER`` or ``"st"``.

    Unknown ``provider_name`` raises ``ValueError`` listing every
    registered provider — add an entry to ``_EMBEDDING_PROVIDERS``, never
    a subclass.
    """
    resolved_provider = (
        provider_name
        or os.environ.get(ENV_PROVIDER)
        or DEFAULT_PROVIDER
    )
    entry = _EMBEDDING_PROVIDERS.get(resolved_provider)
    if entry is None:
        registered = sorted(_EMBEDDING_PROVIDERS.keys())
        raise ValueError(
            f"Unknown embedding provider: {resolved_provider!r}. "
            f"Registered providers: {registered}. Add a registry entry to "
            f"_EMBEDDING_PROVIDERS in lib/embedding/providers.py — no "
            f"subclassing required."
        )

    kind = entry["kind"]

    # Model: explicit arg → env → registry default.
    model_env = entry.get("model_env")
    resolved_model = (
        model_id
        or (os.environ.get(model_env) if model_env else None)
        or entry.get("model_default")
    )
    if not resolved_model:
        raise ValueError(
            f"embedding provider {resolved_provider!r} resolved to an empty "
            f"model_id; set {model_env or ENV_MODEL} or pass model_id="
        )

    # Batch size: env → registry default.
    batch_env = entry.get("batch_size_env")
    batch_default = int(entry.get("batch_size_default", 16))
    batch_size = batch_default
    if batch_env:
        raw_batch = os.environ.get(batch_env)
        if raw_batch:
            try:
                parsed = int(raw_batch)
                if parsed > 0:
                    batch_size = parsed
                else:
                    logger.warning(
                        "%s=%r is not a positive int; using default %d",
                        batch_env, raw_batch, batch_default,
                    )
            except ValueError:
                logger.warning(
                    "%s=%r is not an int; using default %d",
                    batch_env, raw_batch, batch_default,
                )

    device: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    if kind == "st":
        device_env = entry.get("device_env")
        device = (
            (os.environ.get(device_env) if device_env else None)
            or entry.get("device_default")
            or "cpu"
        )
    elif kind == "openai-embeddings":
        base_url_env = entry.get("base_url_env")
        base_url = (
            (os.environ.get(base_url_env) if base_url_env else None)
            or entry.get("base_url_default")
        )
        api_key_env = entry.get("api_key_env")
        api_key = (
            (os.environ.get(api_key_env) if api_key_env else None)
            or entry.get("api_key_default")
        )
        if entry.get("api_key_required", False) and not api_key:
            raise ValueError(
                f"embedding provider {resolved_provider!r} requires "
                f"{api_key_env}; set the environment variable."
            )

    # Per-model metadata (license registry) — trust_remote_code + task prefixes.
    model_meta = EMBEDDING_MODEL_REGISTRY.get(resolved_model, {})
    trust_remote_code = bool(model_meta.get("trust_remote_code", False))
    query_prefix = str(model_meta.get("query_prefix", ""))
    document_prefix = str(model_meta.get("document_prefix", ""))

    return ResolvedEmbeddingProvider(
        provider_name=resolved_provider,
        kind=kind,
        model_id=resolved_model,
        device=device,
        base_url=base_url,
        batch_size=batch_size,
        api_key=api_key,
        trust_remote_code=trust_remote_code,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )


# ---------------------------------------------------------------------------
# Fake deterministic embedder
# ---------------------------------------------------------------------------
def _fake_vector(text: str, dim: int) -> "np.ndarray":
    """Deterministic L2-normalized unit vector for ``text``.

    ``sha256(text)`` seeds a byte stream; consecutive 8-byte chunks
    (re-hashing as needed) map to floats in [-1, 1). The vector is then
    L2-normalized. Same text → byte-identical vector across processes /
    machines (pure stdlib + numpy float64 → float32 cast). Zero network,
    zero weights.
    """
    import numpy as np

    out: List[float] = []
    counter = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{text}\x00{counter}".encode("utf-8")).digest()
        # 32-byte digest → four float64 from 8-byte unsigned ints.
        for i in range(0, len(digest), 8):
            if len(out) >= dim:
                break
            (val,) = struct.unpack(">Q", digest[i : i + 8])
            # Map uint64 → [-1, 1).
            out.append((val / float(1 << 64)) * 2.0 - 1.0)
        counter += 1

    arr = np.asarray(out[:dim], dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        # Degenerate (astronomically unlikely): fall back to a fixed basis.
        arr = np.zeros(dim, dtype=np.float64)
        arr[0] = 1.0
        norm = 1.0
    return (arr / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding client — ONE class, kind-dispatched internally.
# ---------------------------------------------------------------------------
class EmbeddingClient:
    """Encode text → L2-normalized float32 vectors. Kind-dispatched.

    ONE class. NO per-kind subclasses — the ``kind`` field on the
    resolved provider drives an internal dispatch. ``encode_batch`` is the
    single hot path; it returns ``float32 [n, dim]`` with L2-normalized
    rows so cosine similarity reduces to a dot product downstream.

    Honest failure: any backend problem raises
    :class:`EmbeddingBackendUnavailable`. The class never returns a
    fallback vector of any kind.
    """

    def __init__(
        self,
        resolved: ResolvedEmbeddingProvider,
        *,
        offline: bool = False,
        http_client: Optional[Any] = None,
    ) -> None:
        """Build a client for ``resolved``.

        Args:
            resolved: a :class:`ResolvedEmbeddingProvider`.
            offline: ``st`` only — when True (the query-path default),
                forces ``local_files_only=True`` / ``HF_HUB_OFFLINE=1`` so
                a load can never hit the network. Missing cached weights →
                :class:`EmbeddingBackendUnavailable`.
            http_client: ``openai-embeddings`` only — inject an
                ``httpx.Client`` (or a stub with a compatible ``.post``)
                for tests. ``None`` → a real client is built lazily.
        """
        self.resolved = resolved
        self.offline = bool(offline)
        self._http_client = http_client
        self._st_model: Optional[Any] = None
        self._dim: Optional[int] = None
        if resolved.kind == "fake":
            self._dim = int(_EMBEDDING_PROVIDERS["fake"].get("dim", _FAKE_DIM))

    # -- public surface -----------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimension. Probed on first encode for st/openai kinds."""
        if self._dim is None:
            # Probe with a single short string (cheap; result discarded).
            self.encode_batch(["dim probe"])
        assert self._dim is not None
        return self._dim

    def encode_batch(self, texts: List[str]) -> "np.ndarray":
        """Encode ``texts`` → ``float32 [n, dim]`` L2-normalized.

        Empty input returns a ``float32 [0, dim]`` array (dim probed
        lazily for st/openai). Raises
        :class:`EmbeddingBackendUnavailable` on any backend failure.
        """
        import numpy as np

        if not texts:
            if self._dim is None:
                # Probe lazily so an empty first call still settles dim.
                probe = self._encode_batch_impl(["dim probe"])
                self._dim = int(probe.shape[1])
            return np.zeros((0, self._dim), dtype=np.float32)

        out = self._encode_batch_impl(texts)
        if self._dim is None:
            self._dim = int(out.shape[1])
        return out

    def model_fingerprint(self) -> Dict[str, Any]:
        """Replay-identifying metadata for the manifest provenance block."""
        return {
            "model_id": self.resolved.model_id,
            "revision": self._st_revision(),
            "provider_name": self.resolved.provider_name,
            "kind": self.resolved.kind,
            "dim": self._dim,
        }

    # -- internal dispatch --------------------------------------------------

    def _encode_batch_impl(self, texts: List[str]) -> "np.ndarray":
        kind = self.resolved.kind
        if kind == "fake":
            return self._encode_fake(texts)
        if kind == "st":
            return self._encode_st(texts)
        if kind == "openai-embeddings":
            return self._encode_openai(texts)
        raise EmbeddingBackendUnavailable(
            f"unsupported embedding kind: {kind!r}"
        )

    def _encode_fake(self, texts: List[str]) -> "np.ndarray":
        import numpy as np

        dim = int(self._dim or _FAKE_DIM)
        rows = [_fake_vector(t, dim) for t in texts]
        return np.asarray(np.vstack(rows), dtype=np.float32)

    def _ensure_st_model(self) -> Any:
        if self._st_model is not None:
            return self._st_model
        r = self.resolved
        # Offline guard: set the env BEFORE the import + load so HF's hub
        # never reaches out. local_files_only is also passed explicitly.
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise EmbeddingBackendUnavailable(
                    "sentence-transformers is not installed; install the "
                    "[embedding] extra (`pip install -e .[embedding]`) or use "
                    "a different embedding provider. Underlying error: "
                    f"{exc}"
                ) from exc

            kwargs: Dict[str, Any] = {"device": r.device or "cpu"}
            if r.trust_remote_code:
                kwargs["trust_remote_code"] = True
            if self.offline:
                kwargs["local_files_only"] = True
            try:
                self._st_model = SentenceTransformer(r.model_id, **kwargs)
            except TypeError:
                # Older sentence-transformers may not accept local_files_only;
                # the HF_HUB_OFFLINE env already enforces offline mode.
                kwargs.pop("local_files_only", None)
                try:
                    self._st_model = SentenceTransformer(r.model_id, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    raise EmbeddingBackendUnavailable(
                        f"failed to load sentence-transformers model "
                        f"{r.model_id!r} (offline={self.offline}, "
                        f"device={r.device}): {exc}"
                    ) from exc
            except Exception as exc:  # noqa: BLE001 — missing cache, download blocked, etc.
                raise EmbeddingBackendUnavailable(
                    f"failed to load sentence-transformers model "
                    f"{r.model_id!r} (offline={self.offline}, "
                    f"device={r.device}): {exc}"
                ) from exc
        finally:
            if self.offline:
                if prev_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_offline
        return self._st_model

    def _st_revision(self) -> Optional[str]:
        """Best-effort HF revision/commit hash of a loaded st model."""
        model = self._st_model
        if model is None:
            return None
        for attr in ("_model_card_vars", "model_card_data"):
            data = getattr(model, attr, None)
            rev = getattr(data, "base_model_revision", None)
            if isinstance(rev, str) and rev:
                return rev
        return None

    def _encode_st(self, texts: List[str]) -> "np.ndarray":
        import numpy as np

        model = self._ensure_st_model()
        try:
            vecs = model.encode(
                texts,
                batch_size=self.resolved.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 — runtime model failure is unavailable
            raise EmbeddingBackendUnavailable(
                f"sentence-transformers encode failed for model "
                f"{self.resolved.model_id!r}: {exc}"
            ) from exc
        return np.asarray(vecs, dtype=np.float32)

    def _http(self) -> Any:
        if self._http_client is None:
            import httpx

            self._http_client = httpx.Client(timeout=60.0)
        return self._http_client

    def _encode_openai(self, texts: List[str]) -> "np.ndarray":
        """POST {base_url}/embeddings (OpenAI wire shape) against a local server.

        Retry policy mirrors
        ``Trainforge/generators/_openai_compatible_client.DEFAULT_RETRY_STATUS_CODES``
        (429 + 5xx) with bounded exponential backoff. Any unrecoverable
        failure (transport error, non-retryable status, malformed body)
        raises :class:`EmbeddingBackendUnavailable`.
        """
        import time

        import numpy as np

        r = self.resolved
        url = f"{(r.base_url or '').rstrip('/')}/embeddings"
        headers = {"Content-Type": "application/json"}
        if r.api_key:
            headers["Authorization"] = f"Bearer {r.api_key}"
        payload = {"model": r.model_id, "input": list(texts)}

        retry_status = (429, 500, 502, 503, 504)
        max_attempts = 3
        backoff = 1.0
        client = self._http()
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.post(url, json=payload, headers=headers)
            except Exception as exc:  # noqa: BLE001 — transport/connection error
                last_exc = exc
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise EmbeddingBackendUnavailable(
                    f"embedding server unreachable at {url!r} after "
                    f"{attempt} attempts: {exc}"
                ) from exc

            status = getattr(resp, "status_code", 200)
            if status in retry_status and attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            if status >= 400:
                raise EmbeddingBackendUnavailable(
                    f"embedding server at {url!r} returned HTTP {status}"
                )
            try:
                body = resp.json()
                data = body["data"]
                vectors = [row["embedding"] for row in data]
            except Exception as exc:  # noqa: BLE001 — malformed body
                raise EmbeddingBackendUnavailable(
                    f"embedding server at {url!r} returned a malformed body: "
                    f"{exc}"
                ) from exc
            if len(vectors) != len(texts):
                raise EmbeddingBackendUnavailable(
                    f"embedding server returned {len(vectors)} vectors for "
                    f"{len(texts)} inputs"
                )
            arr = np.asarray(vectors, dtype=np.float32)
            # L2-normalize rows (server may or may not have normalized).
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return (arr / norms).astype(np.float32)

        # Exhausted retries on retryable statuses.
        raise EmbeddingBackendUnavailable(
            f"embedding server at {url!r} kept returning retryable errors "
            f"after {max_attempts} attempts"
            + (f": {last_exc}" if last_exc else "")
        )


def build_embedding_client(
    provider_name: Optional[str] = None,
    model_id: Optional[str] = None,
    *,
    offline: bool = False,
    http_client: Optional[Any] = None,
) -> EmbeddingClient:
    """Resolve a provider + construct an :class:`EmbeddingClient`.

    Convenience wrapper over :func:`resolve_embedding_provider` +
    :class:`EmbeddingClient`. The query path calls this with
    ``offline=True`` so a query can never trigger a model download.
    """
    resolved = resolve_embedding_provider(provider_name, model_id)
    return EmbeddingClient(resolved, offline=offline, http_client=http_client)


def allow_fake_enabled() -> bool:
    """Return True when ``ED4ALL_EMBEDDING_ALLOW_FAKE`` is truthy.

    Consumers (the index query path) call this to decide whether a
    ``provider="fake"`` index may be loaded in a production read path.
    Mirrors the ``LOCAL_DISPATCHER_ALLOW_STUB`` opt-in.
    """
    return _env_truthy(ENV_ALLOW_FAKE)


__all__ = [
    "EmbeddingBackendUnavailable",
    "ResolvedEmbeddingProvider",
    "resolve_embedding_provider",
    "EmbeddingClient",
    "build_embedding_client",
    "allow_fake_enabled",
    "EMBEDDING_MODEL_REGISTRY",
]
