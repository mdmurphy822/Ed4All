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
  ``Trainforge/generators/providers/_openai_compatible_client.py::DEFAULT_RETRY_STATUS_CODES``;
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

Device policy: ``ED4ALL_EMBEDDING_DEVICE`` defaults to ``cuda`` for index
builds AND query encoding. CPU remains a fully-supported explicit operator
selection (``ED4ALL_EMBEDDING_DEVICE=cpu``). There is **no** automatic
CUDA→CPU fallback and no ``torch.cuda.is_available()`` probe on this path:
a selected-but-absent device raises :class:`EmbeddingBackendUnavailable`
whose message names the CPU opt-out. Auto-detection is silent degradation,
so ``auto`` is not even a recognized device token.

Public surface (frozen — downstream WS2 executors code against it):

- :class:`EmbeddingBackendUnavailable`
- :class:`ResolvedEmbeddingProvider`
- :func:`resolve_embedding_provider`
- :func:`resolve_embedding_device` / :func:`normalize_device_token`
- :class:`EmbeddingClient`
- :func:`build_embedding_client` (process-level resident client cache)
- :func:`embedding_client_cache_stats`

Benchmark candidates + license metadata: :data:`EMBEDDING_MODEL_REGISTRY`
(all Apache-2.0 / MIT; see ``docs/LICENSING.md`` § "Embedding
providers").
"""
from __future__ import annotations

import hashlib
import logging
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np  # noqa: F401 — type-only

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Truthy-env helper — same convention as
# ``lib/embedding/sentence_embedder.py::_TRUTHY_VALUES``.
# ---------------------------------------------------------------------------
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})

#: Explicit opt-OUT tokens for the flags that default ON. A flag that defaults
#: on cannot use ``_TRUTHY_VALUES`` alone (unset would read as off), so it
#: recognizes an explicit falsey token and treats everything else — including
#: garbage — as the documented default. Same convention as
#: ``SEMANTIK_HEADING_JUDGE`` (default ON, explicit falsey token opts out).
_FALSEY_VALUES = frozenset({"false", "0", "no", "off"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_VALUES


# ---------------------------------------------------------------------------
# Env-var names (single source of truth for the ED4ALL_EMBEDDING_* family).
# Documented in docs/operations/behavior-flags.md (indexed in root CLAUDE.md) + gui/env_catalog.py.
# ---------------------------------------------------------------------------
ENV_PROVIDER = "ED4ALL_EMBEDDING_PROVIDER"
ENV_MODEL = "ED4ALL_EMBEDDING_MODEL"
ENV_BASE_URL = "ED4ALL_EMBEDDING_BASE_URL"
ENV_API_KEY = "ED4ALL_EMBEDDING_API_KEY"
ENV_DEVICE = "ED4ALL_EMBEDDING_DEVICE"
ENV_DTYPE = "ED4ALL_EMBEDDING_DTYPE"
ENV_BATCH_SIZE = "ED4ALL_EMBEDDING_BATCH_SIZE"
ENV_ALLOW_FAKE = "ED4ALL_EMBEDDING_ALLOW_FAKE"

#: Process-level resident embedding-client cache (default ON). ``0`` restores
#: the historical construct-a-client-per-call behavior exactly.
ENV_CLIENT_CACHE = "ED4ALL_EMBEDDING_CLIENT_CACHE"
#: Maximum number of resident clients (ENTRIES, not bytes) held by that cache
#: (LRU, default 2). There is no byte ceiling on the cache — see the cache
#: section below for why, and for what eviction does and does not free.
ENV_CLIENT_CACHE_MAX = "ED4ALL_EMBEDDING_CLIENT_CACHE_MAX"

# --- W1b.1 embed-overflow guard env names -------------------------------
#: Master switch for the over-window chunk guard. **Default ON, and STRICTLY
#: report-only**: it makes the manifest carry the overflow accounting block and
#: nothing else. It never mutates a record, never changes a vector, and never
#: changes ``embeddings_sha256``. An explicit falsey token drops the accounting
#: block. Two behaviors that DO change output are deliberately NOT on this
#: switch: the sub-window SPLIT arm (:data:`ENV_OVERFLOW_SPLIT`) and the
#: encoder ``max_seq_length`` pin (:data:`ENV_MAX_SEQ_PIN`).
ENV_OVERFLOW_GUARD = "ED4ALL_EMBED_OVERFLOW_GUARD"
#: Optional split arm — when the guard is on AND this is truthy, over-window
#: chunks are additionally split into parent-resolving sub-window pieces.
ENV_OVERFLOW_SPLIT = "ED4ALL_EMBED_OVERFLOW_SPLIT"
#: Serving-window token ceiling for the ACCOUNTING arm only (default 512 — the
#: bge / BERT wordpiece window). Changing it re-scores the report; it does NOT
#: touch the encoder. The encoder pin has its own knob, :data:`ENV_MAX_SEQ_PIN`.
ENV_MAX_SEQ_TOKENS = "ED4ALL_EMBED_MAX_SEQ_TOKENS"
#: Encoder ``max_seq_length`` pin — the one knob on this path that CHANGES
#: EMITTED VECTORS, so it is its own explicit opt-in and defaults to unset (no
#: pin, native window untouched). A positive int N pins the loaded ``st`` model
#: to ``min(native, N)``; unset / garbage / non-positive → no pin at all. It is
#: an int rather than a bool precisely because "which window" is the whole
#: decision — an operator arming this is choosing a truncation point on purpose.
ENV_MAX_SEQ_PIN = "ED4ALL_EMBED_MAX_SEQ_PIN"

DEFAULT_PROVIDER = "st"
_FAKE_DIM = 32

#: Encoder precision tokens accepted by ``ED4ALL_EMBEDDING_DTYPE``. ``fp32`` is
#: the default; ``bf16`` is preferred over ``fp16`` for a 1024-dim mean-pooled
#: encoder (same bandwidth win, no loss-scaling or overflow risk on the pooling
#: sum). The value is provenance — it is recorded on the client fingerprint so
#: an index built at one precision is distinguishable from another.
VALID_DTYPES: Tuple[str, ...] = ("fp32", "bf16", "fp16")
DEFAULT_DTYPE = "fp32"

#: Default resident-client cache bound. Unit is ENTRIES (LRU), never bytes.
DEFAULT_CLIENT_CACHE_MAX = 2

#: Default token window for the guard. The st default `BAAI/bge-large-en-v1.5`
#: is a 512-token BERT-family model; the chunker permits ~800-word chunks
#: (`Trainforge.chunker.MAX_CHUNK_SIZE`), so an ~800-word chunk silently
#: overflows the encoder window and its tail is truncated before embedding.
DEFAULT_MAX_SEQ_TOKENS = 512


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
        # Device default is CUDA for index builds AND query encoding (owner
        # decision). CPU stays a fully-supported EXPLICIT operator selection
        # via ED4ALL_EMBEDDING_DEVICE=cpu; there is NO automatic CUDA→CPU
        # fallback anywhere on this path — CUDA selected but unavailable
        # raises EmbeddingBackendUnavailable.
        #
        # DETERMINISM CONTRACT CHANGE (supersedes the old "determinism (D4)"
        # cpu pin): a cuda-built index is NOT bit-reproducible against a
        # cpu-built one — GPU reductions are non-associative — so the
        # byte-identical ``embeddings.npy`` promise now holds only WITHIN a
        # fixed (device, dtype, batch_size) triple. Those three are recorded
        # on ``EmbeddingClient.model_fingerprint()`` precisely so a
        # mixed-device comparison can be refused instead of silently trusted.
        "device_env": ENV_DEVICE,
        "device_default": "cuda",
        # Encoder precision seam, threaded like ``trust_remote_code``: a
        # registry field, never a subclass. Default fp32 → byte-identical to
        # the pre-seam load.
        "dtype_env": ENV_DTYPE,
        "dtype_default": DEFAULT_DTYPE,
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
    re-resolution drift — and so it can serve directly as the key of the
    process-level client cache (:func:`build_embedding_client`). ``device``
    and ``dtype`` are ``st``-only; ``base_url`` / ``api_key`` are
    ``openai-embeddings``-only — each is ``None`` (or the ``dtype``
    default) for the other kinds.
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
    dtype: str = DEFAULT_DTYPE


# ---------------------------------------------------------------------------
# Device / dtype token discipline.
#
# The device token used to be taken VERBATIM from the env and handed to
# ``SentenceTransformer``, so ``auto`` / ``gpu`` / ``CUDA`` / a trailing space
# produced an opaque wrapped load error instead of a config error. Both tokens
# are now normalized and validated at resolution time.
#
# ``auto`` is deliberately NOT a recognized token: auto-detecting a device is
# silent degradation, which this project forbids. The operator names the
# device or takes the documented default.
# ---------------------------------------------------------------------------


def normalize_device_token(raw: Any, *, env_name: str = ENV_DEVICE) -> str:
    """Normalize/validate a torch device token → ``cpu`` | ``cuda`` | ``cuda:N``.

    Case-insensitive after strip. Anything else raises ``ValueError`` naming
    the valid set and the CPU opt-out — never a silent substitution.
    """
    token = str(raw if raw is not None else "").strip().lower()
    if token in ("cpu", "cuda"):
        return token
    if token.startswith("cuda:"):
        index = token.split(":", 1)[1]
        if index.isdigit():
            return f"cuda:{int(index)}"
    raise ValueError(
        f"{env_name}={raw!r} is not a recognized device token. Valid tokens: "
        f"'cpu', 'cuda', 'cuda:N' (case-insensitive). 'auto' is NOT recognized "
        f"— this project never auto-detects a device (auto-detection is silent "
        f"degradation). Set {env_name}=cpu to run on CPU."
    )


def normalize_dtype_token(raw: Any, *, env_name: str = ENV_DTYPE) -> str:
    """Normalize/validate an encoder-precision token → one of :data:`VALID_DTYPES`.

    Case-insensitive after strip. Anything else raises ``ValueError`` naming
    the valid set: a typo'd ``bfloat16`` must not silently encode in fp32 and
    then be recorded as fp32 provenance the operator never asked for.
    """
    token = str(raw if raw is not None else "").strip().lower()
    if token in VALID_DTYPES:
        return token
    raise ValueError(
        f"{env_name}={raw!r} is not a recognized encoder precision. Valid "
        f"tokens: {list(VALID_DTYPES)} (case-insensitive). Prefer 'bf16' over "
        f"'fp16' for a mean-pooled encoder."
    )


def resolve_embedding_device(
    device: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve the embedding device: explicit arg → env → registry default.

    Shared by :func:`resolve_embedding_provider` (the index/query path) and
    ``lib/embedding/sentence_embedder.py`` (the validator tier) so the two
    stacks can never drift onto different DEVICES. Always returns a normalized
    token; an unrecognized token raises ``ValueError``.

    Scope note — the device is the ONLY axis these two stacks share, and this
    function does not make them one stack. They run different MODELS on
    purpose: the index/query path takes the product pin
    ``BAAI/bge-large-en-v1.5`` from :data:`_EMBEDDING_PROVIDERS`, while the
    validator tier takes ``all-MiniLM-L6-v2`` from
    ``sentence_embedder._DEFAULT_MODEL_NAME`` (live — every
    ``try_load_embedder()`` call with no model argument lands on it). Their
    vectors are not interchangeable — different dimensionality, different
    space — so never compare one against the other.
    """
    src = env if env is not None else os.environ
    entry = _EMBEDDING_PROVIDERS["st"]
    raw = (
        device
        or src.get(str(entry.get("device_env") or ENV_DEVICE))
        or entry.get("device_default")
    )
    return normalize_device_token(raw)


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
    dtype: str = DEFAULT_DTYPE

    if kind == "st":
        device_env = entry.get("device_env")
        device = normalize_device_token(
            (os.environ.get(device_env) if device_env else None)
            or entry.get("device_default")
        )
        dtype_env = entry.get("dtype_env")
        dtype = normalize_dtype_token(
            (os.environ.get(dtype_env) if dtype_env else None)
            or entry.get("dtype_default")
            or DEFAULT_DTYPE
        )
        if dtype != DEFAULT_DTYPE and device == "cpu":
            # Half precision on CPU is not a supported configuration for these
            # encoders (no bf16/fp16 kernels worth having, and torch silently
            # upcasts on some builds). Selecting it must FAIL, not be ignored:
            # ignoring it would record a precision in the index provenance
            # that the run never used.
            raise EmbeddingBackendUnavailable(
                f"{ENV_DTYPE}={dtype!r} was selected with {ENV_DEVICE}=cpu. "
                f"Half precision is a CUDA-only encoder setting here; it is "
                f"never silently downgraded to fp32. Either set "
                f"{ENV_DTYPE}=fp32 or select a cuda device."
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
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Fake deterministic embedder
# ---------------------------------------------------------------------------
def _torch_dtype(dtype: str) -> Any:
    """Map a :data:`VALID_DTYPES` token to a ``torch`` dtype.

    ``fp32`` never reaches here (the caller skips the seam entirely, keeping
    the default load byte-identical). A missing torch is a typed
    unavailability, not a silent downgrade to fp32.
    """
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover — torch ships with the extra
        raise EmbeddingBackendUnavailable(
            f"{ENV_DTYPE}={dtype!r} requires torch, which is not installed; "
            f"install the [embedding] extra (`pip install -e .[embedding]`) or "
            f"set {ENV_DTYPE}=fp32. Underlying error: {exc}"
        ) from exc
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16}
    resolved = mapping.get(dtype)
    if resolved is None:  # pragma: no cover — normalize_dtype_token gates this
        raise EmbeddingBackendUnavailable(
            f"{ENV_DTYPE}={dtype!r} has no torch dtype mapping; valid: "
            f"{list(VALID_DTYPES)}"
        )
    return resolved


def _enable_tf32_matmul() -> None:
    """Enable TF32 matmul/cudnn kernels for the half-precision encoder path.

    Only called when a non-fp32 encoder precision was explicitly selected, so
    the default load never changes torch's global precision posture. Logged at
    INFO because the switch is PROCESS-GLOBAL — it also applies to any other
    torch consumer sharing this interpreter (e.g. an in-process NLI head), and
    that is not something to change quietly.
    """
    try:
        import torch  # type: ignore

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as exc:  # noqa: BLE001 — throughput knob, not correctness
        logger.debug("TF32 matmul enable skipped: %s", exc)
        return
    logger.info(
        "%s selected a half-precision encoder; TF32 matmul enabled "
        "PROCESS-GLOBALLY (affects every torch consumer in this interpreter).",
        ENV_DTYPE,
    )


def _st_load_failure_message(
    resolved: ResolvedEmbeddingProvider,
    offline: bool,
    exc: Exception,
) -> str:
    """Operator-actionable message for a failed ``st`` model load.

    Names the CPU opt-out explicitly, mirroring the missing-extras sibling
    above. There is deliberately NO ``torch.cuda.is_available()`` probe and no
    CUDA→CPU downgrade anywhere on this path: a requested device that is not
    there is a hard failure the operator resolves by naming a different
    device, never something this module decides on their behalf.
    """
    return (
        f"failed to load sentence-transformers model {resolved.model_id!r} "
        f"(offline={offline}, device={resolved.device}, "
        f"dtype={resolved.dtype}): {exc}. No automatic CUDA→CPU downgrade is "
        f"performed — set {ENV_DEVICE}=cpu to run this on CPU, or provision "
        f"the requested device."
    )


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
        # Guards the lazy ``st`` model load. Load-bearing since
        # ``build_embedding_client`` began returning a PROCESS-SHARED client:
        # without it, N threads that hit ``encode_batch`` before the first load
        # finishes each construct their own SentenceTransformer (measured: 4
        # threads -> 4 constructions), so the resident-client cache multiplies
        # exactly the allocation it exists to bound — and the process-global
        # ``HF_HUB_OFFLINE`` save/restore inside the load races, which can leave
        # the variable pinned to "1" for the rest of the process.
        self._st_model_lock = threading.Lock()
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
        """Replay-identifying metadata for the manifest provenance block.

        Carries ``device`` / ``batch_size`` / ``dtype`` alongside the model
        identity. Those three are the replay parameters a build is NOT
        reproducible without: with the CUDA default in force, an
        ``embeddings.npy`` is byte-identical only within a fixed
        (device, dtype, batch_size) triple, so an index that does not record
        them cannot be compared against another index at all.

        Every value is read off :attr:`resolved` — never defaulted, never
        substituted. The consumer may therefore treat a missing key as a
        loud failure rather than fabricating ``cpu`` / ``1``.
        """
        device, dtype = self._provenance_device_dtype()
        return {
            "model_id": self.resolved.model_id,
            "revision": self._st_revision(),
            "provider_name": self.resolved.provider_name,
            "kind": self.resolved.kind,
            "dim": self._dim,
            "device": device,
            "batch_size": int(self.resolved.batch_size),
            "dtype": dtype,
        }

    def _provenance_device_dtype(self) -> Tuple[str, str]:
        """(device, dtype) as recorded provenance, honest for every kind.

        ``st`` reports the resolved torch device and encoder precision. The
        other two kinds do not run a local encoder, so reporting a torch
        device would be a fabrication:

        - ``openai-embeddings`` encodes on the REMOTE server, whose device and
          precision this process cannot observe → the ``"server"`` sentinel
          the manifest schema already documents, on both axes.
        - ``fake`` is pure stdlib + numpy float64→float32 arithmetic in this
          process → genuinely ``cpu`` / ``fp32``.
        """
        kind = self.resolved.kind
        if kind == "st":
            return (str(self.resolved.device or ""), str(self.resolved.dtype))
        if kind == "openai-embeddings":
            return ("server", "server")
        return ("cpu", DEFAULT_DTYPE)

    def resident_bytes(self) -> Optional[int]:
        """Bytes held by the loaded encoder, or ``None`` when not knowable.

        Status reporting only (the resident-client cache surfaces it through
        :func:`embedding_client_cache_stats`). ``None`` means "not measured" —
        an unloaded model, a non-``st`` kind, or a torch module that does not
        expose parameters. It is never reported as ``0``, which would read as
        "measured, and free."
        """
        model = self._st_model
        if model is None or self.resolved.kind != "st":
            return None
        try:
            total = 0
            for tensor in list(model.parameters()) + list(model.buffers()):
                total += int(tensor.numel()) * int(tensor.element_size())
            return total
        except Exception as exc:  # noqa: BLE001 — reporting only, never blocks
            logger.debug("resident_bytes probe skipped: %s", exc)
            return None

    def token_counter(self) -> Optional[Callable[[str], int]]:
        """An EXACT tokenizer-backed token counter, or ``None`` when absent.

        The documented ``count_tokens`` seam on :func:`count_overflow_records`
        / :func:`split_overflow_records` defaults to the stdlib
        words × 1.3 :func:`estimate_token_count`. Callers that already hold a
        loaded ``st`` model should pass this instead so the overflow
        accounting is the encoder's real wordpiece count rather than an
        estimate.

        Returns ``None`` (→ caller keeps the documented estimate) when there
        is no local encoder to ask: a non-``st`` kind, a model that has not
        been loaded yet, or a model exposing no usable tokenizer. This is an
        accounting-precision seam, not a quality signal — ``None`` costs
        accuracy in a report, never correctness in a vector.
        """
        if self.resolved.kind != "st":
            return None
        model = self._st_model
        if model is None:
            return None
        tokenizer = getattr(model, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            return None

        def _count(text: str) -> int:
            return len(encode(str(text or ""), add_special_tokens=True))

        try:
            _count("probe")
        except Exception as exc:  # noqa: BLE001 — tokenizer API varies by model
            logger.debug("exact token_counter unavailable: %s", exc)
            return None
        return _count

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
        with self._st_model_lock:
            # Double-checked: a thread that blocked here while another loaded
            # must reuse that model, not build a second one.
            if self._st_model is not None:
                return self._st_model
            return self._load_st_model()

    def _load_st_model(self) -> Any:
        """Construct the ``st`` model. Callers hold ``_st_model_lock``."""
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
            if r.dtype != DEFAULT_DTYPE:
                # Encoder precision seam. bge-large streams ~1.25 GiB of fp32
                # weights per forward on a memory-bandwidth-bound encoder, so
                # halving the weight+activation traffic is the dominant
                # index-build lever — and it is the ENCODER, not the persisted
                # matrix, that pays it (encode_batch still returns float32).
                kwargs["model_kwargs"] = {"torch_dtype": _torch_dtype(r.dtype)}
                _enable_tf32_matmul()
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
                        _st_load_failure_message(r, self.offline, exc)
                    ) from exc
            except Exception as exc:  # noqa: BLE001 — missing cache, download blocked, etc.
                raise EmbeddingBackendUnavailable(
                    _st_load_failure_message(r, self.offline, exc)
                ) from exc
        finally:
            if self.offline:
                if prev_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_offline
        self._pin_max_seq_length(self._st_model)
        return self._st_model

    @staticmethod
    def _pin_max_seq_length(model: Any) -> None:
        """Clamp the loaded encoder's serving window when ``ED4ALL_EMBED_MAX_SEQ_PIN``.

        This is the ONE thing on this path that changes emitted vectors: a
        lowered ``max_seq_length`` truncates input earlier, so the tail of a
        long chunk stops influencing its embedding and ``embeddings_sha256``
        moves. It therefore hangs off its own explicit knob and defaults to
        UNSET — the native window is left exactly as the checkpoint config
        carries it.

        It used to hang off ``ED4ALL_EMBED_OVERFLOW_GUARD``, which is a
        report-only accounting switch that defaults ON. That coupling meant
        arming the accounting silently clamped every encoder with a native
        window wider than the ceiling, contradicting the guard's own promise
        never to change a vector. Splitting them is what makes that promise true.

        ``min(native, pin)`` — the pin only ever LOWERS a window, never raises
        one past what the checkpoint supports. Best-effort: a model without the
        attribute is a no-op.
        """
        pin = resolve_embed_max_seq_pin()
        if pin is None:
            return
        try:
            native = getattr(model, "max_seq_length", None)
            if isinstance(native, int) and native > 0:
                model.max_seq_length = min(native, pin)
            elif native is not None:
                model.max_seq_length = pin
        except Exception as exc:  # noqa: BLE001 — pin is best-effort
            logger.debug("%s max_seq_length pin skipped: %s", ENV_MAX_SEQ_PIN, exc)

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
        ``Trainforge/generators/providers/_openai_compatible_client.DEFAULT_RETRY_STATUS_CODES``
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


# ---------------------------------------------------------------------------
# Process-level resident client cache.
#
# ``build_embedding_client`` historically constructed a BRAND-NEW client on
# every call, and the query path calls it once per query — so every query
# re-ran the model load. Measured CPU-pinned: ~0.48 s warm (3.67 s cold) per
# query, and the library-wide path pays it once per indexed course because it
# loops courses serially. With the CUDA default in force it is worse than slow:
# an unmemoized client allocates and frees ~1.25 GiB of the SAME unified memory
# pool a co-resident inference seat sizes its KV cache against, on every query.
#
# Mirrors the proven singleton at
# ``lib/embedding/sentence_embedder.py::_EMBEDDER_CACHE`` — module-level dict,
# one lock, double-checked insert — rather than inventing a second pattern.
#
# The key is the frozen :class:`ResolvedEmbeddingProvider` itself plus
# ``offline``. That is a strict superset of (provider, model, device, dtype,
# batch_size, offline): it additionally separates base_url / api_key /
# trust_remote_code / task prefixes, so no two differently-configured clients
# can ever alias.
#
# BOUND: entries, NOT bytes. Stated plainly because the distinction matters —
# ``ED4ALL_EMBEDDING_CLIENT_CACHE_MAX`` caps how many clients are resident, and
# there is NO byte ceiling anywhere on this cache. Two entries of a 1024-dim
# encoder and two entries of an 8B one are the same to it. Nor is eviction a
# free: ``popitem`` drops this dict's reference and nothing else — the evicted
# client is still alive for as long as any caller holds it (and
# ``build_embedding_client`` hands out SHARED clients, so callers routinely
# do), and even the last reference dying returns CUDA blocks to torch's caching
# allocator rather than to the OS. Eviction bounds this dict; it does not free
# weights on demand.
#
# A real byte bound was considered and is not implementable here: the encoder
# load is deferred to the first encode, so an entry's size is unknown at insert
# time (``resident_bytes()`` returns None until then), and a byte-capped cache
# that cannot free what it evicts would enforce nothing anyway. What IS honest
# is measurement: ``embedding_client_cache_stats()`` reports per-entry
# ``resident_bytes`` (``None`` = not measured, never a fabricated 0) plus
# ``max_entries``, so an operator can see the real footprint instead of
# trusting a cap that does not exist.
#
# This cache holds CLIENTS only. It deliberately does NOT cache the index or
# the matrix: the only freshness guarantee on the retrieval path is that
# ``load_vector_index`` re-hashes the on-disk artifacts per query, and caching
# those would remove it (a rebuild-then-query in one process would serve the
# pre-rebuild matrix).
# ---------------------------------------------------------------------------
_CLIENT_CACHE: "OrderedDict[Tuple[Any, ...], EmbeddingClient]" = OrderedDict()
_CLIENT_CACHE_LOCK = threading.Lock()


def resolve_client_cache_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True unless ``ED4ALL_EMBEDDING_CLIENT_CACHE`` is an explicit falsey token.

    Default ON. ``0`` / ``false`` / ``no`` / ``off`` restores the historical
    construct-a-client-per-call behavior exactly; anything else (including
    garbage) keeps the documented default.
    """
    src = env if env is not None else os.environ
    return str(src.get(ENV_CLIENT_CACHE, "")).strip().lower() not in _FALSEY_VALUES


def resolve_client_cache_max(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve the resident-client LRU bound in ENTRIES (parse-with-fallback → 2, min 1).

    Entries, not bytes: this caps how many clients stay resident, not how much
    memory they occupy. A cache holding ``max_entries`` clients of a large
    encoder is unbounded in bytes. Read ``resident_bytes`` off
    :func:`embedding_client_cache_stats` for the measured footprint.
    """
    src = env if env is not None else os.environ
    raw = src.get(ENV_CLIENT_CACHE_MAX)
    if raw is None:
        return DEFAULT_CLIENT_CACHE_MAX
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not an int; using default %d",
            ENV_CLIENT_CACHE_MAX, raw, DEFAULT_CLIENT_CACHE_MAX,
        )
        return DEFAULT_CLIENT_CACHE_MAX
    return value if value >= 1 else DEFAULT_CLIENT_CACHE_MAX


def embedding_client_cache_stats() -> Dict[str, Any]:
    """Status snapshot of the resident-client cache (reporting only).

    ``resident_bytes`` sums :meth:`EmbeddingClient.resident_bytes` over the
    entries that can report it and is ``None`` when none can — "not measured",
    never a fabricated ``0``. ``unmeasured_entries`` says how many entries were
    skipped so the sum is never mistaken for a total.
    """
    with _CLIENT_CACHE_LOCK:
        entries = list(_CLIENT_CACHE.items())
    measured = 0
    unmeasured = 0
    total: Optional[int] = None
    described: List[Dict[str, Any]] = []
    for (resolved, offline), client in entries:
        size = client.resident_bytes()
        if size is None:
            unmeasured += 1
        else:
            measured += 1
            total = size if total is None else total + size
        described.append(
            {
                "provider_name": resolved.provider_name,
                "model_id": resolved.model_id,
                "device": resolved.device,
                "dtype": resolved.dtype,
                "batch_size": resolved.batch_size,
                "offline": offline,
                "resident_bytes": size,
            }
        )
    return {
        "enabled": resolve_client_cache_enabled(),
        "max_entries": resolve_client_cache_max(),
        "entries": len(entries),
        "measured_entries": measured,
        "unmeasured_entries": unmeasured,
        "resident_bytes": total,
        "clients": described,
    }


def _reset_embedding_client_cache_for_tests() -> None:
    """Test-only seam — drop every resident client."""
    with _CLIENT_CACHE_LOCK:
        _CLIENT_CACHE.clear()


def build_embedding_client(
    provider_name: Optional[str] = None,
    model_id: Optional[str] = None,
    *,
    offline: bool = False,
    http_client: Optional[Any] = None,
) -> EmbeddingClient:
    """Resolve a provider + return a (by default resident) :class:`EmbeddingClient`.

    Convenience wrapper over :func:`resolve_embedding_provider` +
    :class:`EmbeddingClient`. The query path calls this with
    ``offline=True`` so a query can never trigger a model download.

    Resolution ALWAYS runs — the env is re-read on every call, so a changed
    device / model / batch size lands on a different cache entry rather than
    being served stale. Only the constructed client is reused, and only when
    every resolved field plus ``offline`` matches. An injected ``http_client``
    is caller-owned state and is never cached.

    Returned clients are shared across callers in the process. That is the
    same sharing contract ``try_load_embedder`` already relies on: callers
    read only, and ``encode_batch`` is safe for these consumers.
    """
    resolved = resolve_embedding_provider(provider_name, model_id)
    if http_client is not None or not resolve_client_cache_enabled():
        return EmbeddingClient(resolved, offline=offline, http_client=http_client)

    key = (resolved, bool(offline))
    with _CLIENT_CACHE_LOCK:
        cached = _CLIENT_CACHE.get(key)
        if cached is not None:
            _CLIENT_CACHE.move_to_end(key)
            return cached
        # Construction is cheap — the encoder load stays deferred to the first
        # encode — so holding the lock across it costs nothing.
        client = EmbeddingClient(resolved, offline=offline)
        _CLIENT_CACHE[key] = client
        max_entries = resolve_client_cache_max()
        while len(_CLIENT_CACHE) > max_entries:
            evicted_key, _ = _CLIENT_CACHE.popitem(last=False)
            logger.debug(
                "embedding client cache evicted %s (bound %d)",
                evicted_key[0].model_id, max_entries,
            )
        return client


def allow_fake_enabled() -> bool:
    """Return True when ``ED4ALL_EMBEDDING_ALLOW_FAKE`` is truthy.

    Consumers (the index query path) call this to decide whether a
    ``provider="fake"`` index may be loaded in a production read path.
    Mirrors the ``LOCAL_DISPATCHER_ALLOW_STUB`` opt-in.
    """
    return _env_truthy(ENV_ALLOW_FAKE)


# ---------------------------------------------------------------------------
# W1b.1 — embed-overflow guard. THREE independent arms, three knobs.
#
# The chunker emits chunks up to ~800 words (``Trainforge.chunker.
# MAX_CHUNK_SIZE``) but the st default ``BAAI/bge-large-en-v1.5`` is a
# 512-token wordpiece model, so a long chunk silently truncates before it is
# embedded — its tail never influences the vector and can never be retrieved.
#
#   (a) COUNT   — ED4ALL_EMBED_OVERFLOW_GUARD, default ON. Counts over-window
#                 chunks into the index manifest so the defect is observable.
#                 Pure accounting: mutates nothing, changes no vector, changes
#                 no ``embeddings_sha256``. Scored against
#                 ED4ALL_EMBED_MAX_SEQ_TOKENS.
#   (b) SPLIT   — ED4ALL_EMBED_OVERFLOW_SPLIT, default OFF. Splits over-window
#                 chunks into parent-resolving sub-window pieces
#                 (anti-fabrication: every sub-id resolves back to its parent
#                 and every sub-text is a strict contiguous word-slice of the
#                 parent). Changes chunk identity and the chunkset hash.
#   (c) PIN     — ED4ALL_EMBED_MAX_SEQ_PIN, default UNSET. Clamps the loaded
#                 encoder's ``max_seq_length``. Changes emitted vectors.
#
# (a) is safe to leave on because it only reports. (b) and (c) each change
# output, so each is its own opt-in — arming the report must never silently
# arm a mutation. All resolvers are parse-with-fallback (garbage → the safe
# default, which for (b) and (c) is "do not touch the output").
# ---------------------------------------------------------------------------


def resolve_embed_overflow_guard(env: Optional[Dict[str, str]] = None) -> bool:
    """Return True unless ``ED4ALL_EMBED_OVERFLOW_GUARD`` is explicitly falsey.

    **Default ON, and this arm ONLY counts.** Over-window truncation is silent
    by construction — a chunk longer than the encoder's serving window loses
    its tail before it is ever embedded, and nothing downstream can tell. With
    the production pin's 512-token window against a chunker that permits
    ~800-word chunks, that is a large, permanently-invisible share of the
    corpus unless it is counted. Counting is pure accounting: it never mutates
    a record, never changes a vector, and never changes ``embeddings_sha256``.

    That claim is now literally true. Two output-changing arms are gated
    separately, and neither is reachable from this switch:

    - :func:`resolve_embed_overflow_split` (default OFF) — splitting changes
      chunk identity and the chunkset hash.
    - :func:`resolve_embed_max_seq_pin` (default UNSET) — clamping the
      encoder's ``max_seq_length`` changes emitted vectors. This function used
      to gate that pin too, which is exactly how a report-only default-ON
      switch acquired the power to silently re-embed a corpus.

    Explicit ``0`` / ``false`` / ``no`` / ``off`` opts out; anything else
    (including garbage) keeps the documented default.
    """
    src = env if env is not None else os.environ
    return str(src.get(ENV_OVERFLOW_GUARD, "")).strip().lower() not in _FALSEY_VALUES


def resolve_embed_overflow_split(env: Optional[Dict[str, str]] = None) -> bool:
    """Return True when ``ED4ALL_EMBED_OVERFLOW_SPLIT`` is truthy (default OFF).

    The split arm is a no-op unless :func:`resolve_embed_overflow_guard` is
    also on — the master switch gates the whole feature.
    """
    src = env if env is not None else os.environ
    return src.get(ENV_OVERFLOW_SPLIT, "").strip().lower() in _TRUTHY_VALUES


def resolve_embed_max_seq_tokens(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve the ACCOUNTING serving-window ceiling (parse-with-fallback → 512).

    Scores the report arm only — which chunks get counted as over-window. It
    does NOT reach the encoder: lowering it re-scores the manifest block and
    leaves every emitted vector byte-identical. The encoder-side clamp is
    :func:`resolve_embed_max_seq_pin`, a separate knob on purpose, so an
    operator tightening the report cannot accidentally re-embed the corpus.

    Garbage / non-integer / non-positive values fall back to
    :data:`DEFAULT_MAX_SEQ_TOKENS` (mirrors ``ED4ALL_ANSWER_NUM_CTX``).
    """
    src = env if env is not None else os.environ
    raw = src.get(ENV_MAX_SEQ_TOKENS)
    if raw is None:
        return DEFAULT_MAX_SEQ_TOKENS
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_SEQ_TOKENS
    return val if val > 0 else DEFAULT_MAX_SEQ_TOKENS


def resolve_embed_max_seq_pin(env: Optional[Dict[str, str]] = None) -> Optional[int]:
    """Resolve the encoder ``max_seq_length`` pin, or ``None`` for "no pin".

    ``None`` (the default) means the loaded model keeps whatever window its
    checkpoint config carries — the encoder is not touched at all, so emitted
    vectors are exactly what the model would produce unpinned. A positive int
    N arms the clamp at ``min(native, N)``.

    Parse-with-fallback, and the fallback is deliberately "no pin": garbage, a
    non-integer, or a non-positive value must not be silently rounded up into
    *some* clamp, because every clamp value is a different corpus. Unset is the
    only default that leaves ``embeddings_sha256`` alone.

    Note for the product pin: ``BAAI/bge-large-en-v1.5`` has a native
    ``max_seq_length`` of 512, so ``ED4ALL_EMBED_MAX_SEQ_PIN=512`` is a
    verified no-op on it and only bites models with a wider native window.
    Anything BELOW 512 does change the product model's vectors, which is
    precisely why this cannot ride on an accounting flag.
    """
    src = env if env is not None else os.environ
    raw = src.get(ENV_MAX_SEQ_PIN)
    if raw is None:
        return None
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def estimate_token_count(text: str) -> int:
    """Cheap upper-ish estimate of wordpiece tokens for ``text``.

    No model load — a whitespace word count scaled by a fixed 1.3
    words→subword-tokens factor (BERT-family wordpiece averages ~1.3 tokens
    per whitespace word on English prose). Deliberately deterministic +
    stdlib-only so the overflow accounting is replayable and the manifest
    count is stable across machines. Consumers wanting exact tokenization
    inject a real ``count_tokens`` callable (e.g. ``model.tokenizer``).
    """
    words = len((text or "").split())
    if words == 0:
        return 0
    return int(words * 1.3 + 0.5)


def _resolve_counter(
    count_tokens: Optional[Callable[[str], int]],
) -> Callable[[str], int]:
    return count_tokens if count_tokens is not None else estimate_token_count


def count_overflow_records(
    records: List[Dict[str, Any]],
    max_tokens: int,
    *,
    count_tokens: Optional[Callable[[str], int]] = None,
    id_key: str = "id",
    text_key: str = "text",
    sample_limit: int = 10,
) -> Dict[str, Any]:
    """Count records whose ``text`` exceeds ``max_tokens`` in the encoder window.

    Pure accounting — never mutates ``records``. Returns a manifest-ready
    block:

        {"max_seq_tokens", "records_scanned", "overflow_count",
         "overflow_rate", "max_observed_tokens", "overflow_chunk_ids"}

    ``count_tokens`` defaults to :func:`estimate_token_count`; tests inject a
    stub (e.g. ``str.split`` length) so the logic is exercised with zero model
    weights. ``overflow_chunk_ids`` is capped at ``sample_limit`` (post-hoc
    triage sample, not a full dump).
    """
    counter = _resolve_counter(count_tokens)
    overflow_ids: List[str] = []
    overflow_count = 0
    max_observed = 0
    scanned = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        scanned += 1
        toks = counter(str(rec.get(text_key, "") or ""))
        if toks > max_observed:
            max_observed = toks
        if toks > max_tokens:
            overflow_count += 1
            if len(overflow_ids) < sample_limit:
                cid = rec.get(id_key) or rec.get("chunk_id")
                overflow_ids.append(str(cid) if cid is not None else "")
    return {
        "max_seq_tokens": int(max_tokens),
        "records_scanned": scanned,
        "overflow_count": overflow_count,
        "overflow_rate": (overflow_count / scanned) if scanned else 0.0,
        "max_observed_tokens": max_observed,
        "overflow_chunk_ids": overflow_ids,
    }


def split_overflow_records(
    records: List[Dict[str, Any]],
    max_tokens: int,
    *,
    count_tokens: Optional[Callable[[str], int]] = None,
    id_key: str = "id",
    text_key: str = "text",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Split over-window records into parent-resolving sub-window pieces.

    Anti-fabrication contract:

    - Every sub-piece's ``text`` is a STRICT contiguous whitespace-word slice
      of the parent's text — never synthesized, never reworded. Concatenating a
      parent's sub-pieces in order reproduces the parent word sequence.
    - Every sub-piece carries ``parent_chunk_id`` == the parent's id and an id
      of the form ``f"{parent_id}#p{n}"`` (n = 0-based), so a sub-id always
      resolves back to a real parent chunk.
    - Non-overflow records pass through UNCHANGED (same object identity), so a
      corpus with no over-window chunk is byte-identical.

    Returns ``(new_records, stats)`` where stats carries ``overflow_count`` and
    ``sub_pieces_created``. ``count_tokens`` defaults to
    :func:`estimate_token_count`; a stub is injected in tests.
    """
    counter = _resolve_counter(count_tokens)
    out: List[Dict[str, Any]] = []
    overflow_count = 0
    sub_pieces = 0
    for rec in records:
        if not isinstance(rec, dict):
            out.append(rec)
            continue
        text = str(rec.get(text_key, "") or "")
        if counter(text) <= max_tokens:
            out.append(rec)
            continue
        words = text.split()
        pieces = _slice_words_to_window(words, max_tokens, counter)
        if len(pieces) <= 1:
            # Degenerate (a single word already over-window, or empty) —
            # keep the original whole rather than emit a lone identical piece.
            out.append(rec)
            continue
        overflow_count += 1
        parent_id = rec.get(id_key) or rec.get("chunk_id")
        parent_id = str(parent_id) if parent_id is not None else ""
        for n, piece_words in enumerate(pieces):
            sub = dict(rec)
            sub[text_key] = " ".join(piece_words)
            sub[id_key] = f"{parent_id}#p{n}"
            sub["parent_chunk_id"] = parent_id
            sub["overflow_split"] = True
            out.append(sub)
            sub_pieces += 1
    stats = {
        "max_seq_tokens": int(max_tokens),
        "overflow_count": overflow_count,
        "sub_pieces_created": sub_pieces,
    }
    return out, stats


def _slice_words_to_window(
    words: List[str],
    max_tokens: int,
    counter: Callable[[str], int],
) -> List[List[str]]:
    """Greedily pack ``words`` into contiguous windows each ≤ ``max_tokens``.

    A single word wider than the window is emitted alone (it cannot be split
    without corrupting content — anti-fabrication). Deterministic.
    """
    windows: List[List[str]] = []
    current: List[str] = []
    for word in words:
        candidate = current + [word]
        if current and counter(" ".join(candidate)) > max_tokens:
            windows.append(current)
            current = [word]
        else:
            current = candidate
    if current:
        windows.append(current)
    return windows


__all__ = [
    "EmbeddingBackendUnavailable",
    "ResolvedEmbeddingProvider",
    "resolve_embedding_provider",
    "resolve_embedding_device",
    "normalize_device_token",
    "normalize_dtype_token",
    "EmbeddingClient",
    "build_embedding_client",
    "embedding_client_cache_stats",
    "resolve_client_cache_enabled",
    "resolve_client_cache_max",
    "allow_fake_enabled",
    "EMBEDDING_MODEL_REGISTRY",
    "VALID_DTYPES",
    "DEFAULT_DTYPE",
    "DEFAULT_CLIENT_CACHE_MAX",
    "resolve_embed_overflow_guard",
    "resolve_embed_overflow_split",
    "resolve_embed_max_seq_tokens",
    "resolve_embed_max_seq_pin",
    "estimate_token_count",
    "count_overflow_records",
    "split_overflow_records",
    "DEFAULT_MAX_SEQ_TOKENS",
]
