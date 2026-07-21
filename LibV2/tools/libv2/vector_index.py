"""On-device vector index — pure-numpy exact-search build/read/query.

WS2 Executor E2. Builds and reads the per-course vector index that backs
real semantic retrieval (no FAISS, no sqlite-vec — exact cosine over
L2-normalized float32 in numpy; the measured corpus scale is <=3K
vectors / ~12 MB worst case, where exact search is sub-millisecond AND
the determinism-safest option).

Artifact layout (``LibV2/courses/<slug>/vector_index/``)::

    embeddings.npy   float32 [N, dim], C-order, L2-normalized rows;
                     row i corresponds to id_map.chunk_ids[i].
    id_map.json      {"schema_version": "1.0", "chunk_ids": [...]} — order
                     is load-bearing (it is the row order of embeddings.npy).
    manifest.json    provenance + integrity manifest
                     (schemas/library/vector_index_manifest.schema.json).

Determinism contract (D4): same machine + venv + provider + model +
device=cpu + batch_size => ``embeddings.npy`` and ``id_map.json`` are
byte-identical across rebuilds; ``manifest.json`` is identical modulo the
optional ``generated_at`` field, which is the ONLY non-deterministic
section and is excluded from every content hash. Knobs that make this
hold: sorted/stable chunk iteration (we preserve chunks.jsonl line order,
which the chunker emits deterministically), pinned float32 dtype, C-order
arrays, ``np.save`` with ``allow_pickle=False``, and a canonical
``json.dumps(..., sort_keys=True)`` for id_map / manifest.

Anti-silent-degradation (D7): the read path raises typed, honest errors
and NEVER falls back to lexical/BM25 results:

* :class:`SemanticIndexMissing` — no ``vector_index/`` for the course.
* :class:`SemanticIndexStale` — ``source_chunks_sha256`` no longer matches
  the live chunkset, or an on-disk sha/count mismatch, or a schema-invalid
  manifest.
* :class:`FakeIndexRefused` — the manifest's provider is the test ``fake``
  provider and ``ED4ALL_EMBEDDING_ALLOW_FAKE`` is not set
  (anti-production-poisoning; mirrors ``LOCAL_DISPATCHER_ALLOW_STUB``).

``EmbeddingBackendUnavailable`` (weights/server absent) is owned by
``lib.embedding.providers`` and re-raised from the query/build path; it is
imported lazily so this module has no import-time coupling to E1's module
— the build API is duck-typed against the frozen ``EmbeddingClient``
protocol (``encode_batch`` / ``dim`` / ``model_fingerprint``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lib.libv2_storage import (
    DIRNAME_TO_CHUNKSET_KIND as _DIRNAME_TO_KIND,
    VECTOR_INDEX_DIRNAME,
    VECTOR_INDEX_MANIFEST_FILENAME as MANIFEST_FILENAME,
    resolve_imscc_chunks_dir,
)
from lib.generation.stop_control import check_stop
from lib.utils.hashing import sha256_file, sha256_text

try:  # Trainforge is a sibling package; resolved at runtime in-repo.
    from Trainforge.chunker import CHUNKER_SCHEMA_VERSION as _CHUNKER_VERSION
except Exception:  # noqa: BLE001 — keep the index buildable in slim installs.
    _CHUNKER_VERSION = "v4"


__all__ = [
    "VECTOR_INDEX_DIRNAME",
    "EMBEDDINGS_FILENAME",
    "ID_MAP_FILENAME",
    "MANIFEST_FILENAME",
    "SemanticIndexError",
    "SemanticIndexMissing",
    "SemanticIndexStale",
    "FakeIndexRefused",
    "SemanticModelMismatch",
    "VectorIndexManifest",
    "VectorIndex",
    "build_vector_index",
    "load_vector_index",
    "resolve_chunks_for_index",
]


# VECTOR_INDEX_DIRNAME / MANIFEST_FILENAME are imported above from
# lib.libv2_storage (numpy-free single source of truth) so the pure-lexical /
# engine="auto" code paths can probe index presence without importing numpy.
EMBEDDINGS_FILENAME = "embeddings.npy"
ID_MAP_FILENAME = "id_map.json"

_ID_MAP_SCHEMA_VERSION = "1.0"
_MANIFEST_SCHEMA_VERSION = "1.0"
_INDEX_TYPE = "exact-numpy"
_FAKE_PROVIDER = "fake"

# chunkset-dir name -> manifest chunkset_kind discriminator. Single source of
# truth lives in ``lib.libv2_storage.DIRNAME_TO_CHUNKSET_KIND`` (imported above
# as ``_DIRNAME_TO_KIND``) so the build path and the query path
# (``resolve_chunks_path_for_query``) can never disagree on the mapping.

# Float32 epsilon floor used when L2-normalizing so an all-zero embedding
# row (degenerate, but possible from a misbehaving provider) does not
# divide-by-zero — it stays the zero vector and simply never ranks.
_NORM_EPS = 1e-12


# --------------------------------------------------------------------------
# Typed errors (D7 — anti-silent-degradation; never caught into BM25)
# --------------------------------------------------------------------------


class SemanticIndexError(RuntimeError):
    """Base class for every typed vector-index read failure."""


class SemanticIndexMissing(SemanticIndexError):
    """No ``vector_index/`` artifact for the course."""


class SemanticIndexStale(SemanticIndexError):
    """The index no longer matches the live chunkset / on-disk bytes."""


class FakeIndexRefused(SemanticIndexError):
    """Manifest provider is ``fake`` and fake indexes are not allowed."""


class SemanticModelMismatch(SemanticIndexError):
    """A query-side embedding client does not match the index's manifest.

    Raised when an explicit ``EmbeddingClient`` is passed whose
    ``model_fingerprint()["model_id"]`` differs from the index manifest's
    ``embedding_model_id`` — querying a foreign-model index produces
    silently meaningless cosines, so we fail closed rather than serve them.
    """


# --------------------------------------------------------------------------
# Manifest dataclass (frozen signature, § 3)
# --------------------------------------------------------------------------


@dataclass
class VectorIndexManifest:
    schema_version: str
    embedding_provider: str
    embedding_kind: str
    embedding_model_id: str
    embedding_model_revision: Optional[str]
    embedding_dim: int
    normalized: bool
    index_type: str
    chunker_version: str
    chunkset_kind: str
    source_chunks_sha256: str
    embeddings_sha256: str
    id_map_sha256: str
    chunks_count: int
    text_field_policy: str
    batch_size: int
    device: str
    # Asymmetric retrieval prefixes (D6). Recorded for replay/audit: passages
    # were embedded with ``document_prefix`` prepended at build time; queries
    # get ``query_prefix`` at search time. Empty string for symmetric models.
    # Default "" keeps pre-prefix manifests loadable (back-compat).
    document_prefix: str = ""
    query_prefix: str = ""
    # W1b.2 (opt-in ED4ALL_EMBED_OVERFLOW_GUARD): over-window accounting block
    # (count_overflow_records output — max_seq_tokens / overflow_count /
    # overflow_rate / max_observed_tokens / overflow_chunk_ids). Omitted when
    # None (guard off) so the off-path manifest is byte-identical; it IS part of
    # the content hash when present (real build provenance).
    embed_overflow: Optional[Dict[str, Any]] = None
    generated_at: Optional[str] = None

    # Field order used for the on-disk dict + the determinism contract.
    # ``generated_at`` is appended last and is the ONLY field excluded from
    # the content hash. ``embed_overflow`` is omitted when None (like
    # generated_at) so a guard-off build is byte-identical.
    _ORDER: Tuple[str, ...] = (
        "schema_version",
        "embedding_provider",
        "embedding_kind",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_dim",
        "normalized",
        "index_type",
        "chunker_version",
        "chunkset_kind",
        "source_chunks_sha256",
        "embeddings_sha256",
        "id_map_sha256",
        "chunks_count",
        "text_field_policy",
        "batch_size",
        "device",
        "document_prefix",
        "query_prefix",
        "embed_overflow",
        "generated_at",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorIndexManifest":
        known = {f for f in cls._ORDER}
        kwargs = {k: data.get(k) for k in known if k in data}
        # Required positional fields validated by the caller / validator;
        # here we just construct, defaulting generated_at to None.
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_file(cls, path: Path) -> "VectorIndexManifest":
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise SemanticIndexStale(
                f"vector-index manifest root is not a JSON object: {path}"
            )
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Ordered dict in canonical field order; ``generated_at`` omitted
        when None (keeps the determinism contract crisp)."""
        out: Dict[str, Any] = {}
        for key in self._ORDER:
            if key == "generated_at" and self.generated_at is None:
                continue
            if key == "embed_overflow" and self.embed_overflow is None:
                continue
            out[key] = getattr(self, key)
        return out

    def content_dict(self) -> dict:
        """The determinism-contracted subset (everything except
        ``generated_at``). Two builds over identical inputs produce
        identical ``content_dict()`` output."""
        out = self.to_dict()
        out.pop("generated_at", None)
        return out


# --------------------------------------------------------------------------
# Read-side index object (frozen signature, § 3)
# --------------------------------------------------------------------------


class VectorIndex:
    """Loaded, query-ready exact-search index."""

    def __init__(
        self,
        manifest: VectorIndexManifest,
        chunk_ids: List[str],
        matrix: np.ndarray,
    ) -> None:
        self.manifest = manifest
        self.chunk_ids = chunk_ids
        self.matrix = matrix

    def search(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """Exact cosine search (dot product over unit vectors).

        Returns ``[(chunk_id, score), ...]`` sorted by score descending.
        Ties are broken by ``chunk_id`` ascending so the ordering is
        fully deterministic (D2/D4). ``score`` is the cosine similarity in
        ``[-1, 1]`` — NOT comparable to BM25 scores.
        """
        if top_k <= 0 or self.matrix.shape[0] == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} != index dim {self.matrix.shape[1]}"
            )
        # Normalize the query so the dot product is a true cosine even if
        # the caller passed an un-normalized vector.
        qn = float(np.linalg.norm(q))
        if qn > _NORM_EPS:
            q = q / qn

        scores = self.matrix @ q  # [N], float32

        k = min(top_k, scores.shape[0])
        # argsort is ascending + stable; negate for descending, then apply
        # the chunk_id tiebreak explicitly so equal scores order by id asc.
        order = sorted(
            range(scores.shape[0]),
            key=lambda i: (-float(scores[i]), self.chunk_ids[i]),
        )
        return [(self.chunk_ids[i], float(scores[i])) for i in order[:k]]


# --------------------------------------------------------------------------
# Chunkset resolution + text extraction
# --------------------------------------------------------------------------


def resolve_chunks_for_index(
    course_dir: Path, chunkset: Optional[str] = None
) -> Tuple[Path, str]:
    """Resolve ``(chunks_jsonl_path, chunkset_kind)`` for a course.

    ``chunkset`` pins a specific dir name (``imscc`` | ``dart`` |
    ``corpus-legacy``); ``None`` defers to the canonical resolver
    precedence (imscc_chunks -> semantik_chunks -> dart_chunks -> legacy
    corpus) keyed on the presence of ``chunks.jsonl`` so an empty scaffold
    dir doesn't win. (``dart``/``corpus-legacy`` are legacy read-only pins.)
    """
    course_dir = Path(course_dir)
    if chunkset:
        kind_to_dir = {v: k for k, v in _DIRNAME_TO_KIND.items()}
        # Accept both the manifest kind ('imscc') and the raw dir name
        # ('imscc_chunks') as the pin value.
        if chunkset in kind_to_dir:
            dirname = kind_to_dir[chunkset]
        elif chunkset in _DIRNAME_TO_KIND:
            dirname = chunkset
        else:
            raise ValueError(
                f"unknown chunkset pin {chunkset!r}; expected one of "
                f"{sorted(set(_DIRNAME_TO_KIND.values()))} or a chunkset "
                f"dir name {sorted(_DIRNAME_TO_KIND)}"
            )
        chunks_dir = course_dir / dirname
        kind = _DIRNAME_TO_KIND[dirname]
    else:
        chunks_dir = resolve_imscc_chunks_dir(
            course_dir, filename="chunks.jsonl"
        )
        kind = _DIRNAME_TO_KIND.get(chunks_dir.name, chunks_dir.name)
    return chunks_dir / "chunks.jsonl", kind


def _iter_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Read chunks.jsonl in file order (load-bearing — it is the row
    order of embeddings.npy). One JSON object per non-empty line."""
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            chunks.append(json.loads(raw))
    return chunks


def _chunk_id(chunk: Dict[str, Any]) -> str:
    """Stable chunk identifier. Chunks emit ``id`` (canonical) but some
    fixtures / legacy rows use ``chunk_id`` — accept both."""
    cid = chunk.get("id") or chunk.get("chunk_id")
    if not cid:
        raise ValueError(
            "chunk is missing both 'id' and 'chunk_id'; cannot build index"
        )
    return str(cid)


def _embed_text(
    chunk: Dict[str, Any],
    text_field_policy: str,
    document_prefix: str = "",
) -> str:
    """Build the string embedded for a chunk per ``text_field_policy``
    (D6). 'text+heading' prepends the section heading (the most
    signal-dense string per the Wave-84 finding) when present.

    ``document_prefix`` (e.g. nomic's ``"search_document: "``) is prepended
    to the FINAL passage string when non-empty — asymmetric-retrieval models
    embed passages and queries under distinct task prefixes; the matching
    query prefix is applied query-side in
    ``semantic_retriever._embed_query``. Empty for symmetric models (and for
    bge, whose instruction prefix is query-side only)."""
    text = str(chunk.get("text") or "")
    if text_field_policy == "text":
        body = text
    elif text_field_policy == "retrieval_text":
        rt = chunk.get("retrieval_text")
        body = str(rt) if rt else text
    else:
        # default: "text+heading"
        source = chunk.get("source")
        heading = ""
        if isinstance(source, dict):
            heading = str(source.get("section_heading") or "")
        body = f"{heading}\n{text}" if heading else text
    return f"{document_prefix}{body}" if document_prefix else body


# --------------------------------------------------------------------------
# Deterministic serialization helpers
# --------------------------------------------------------------------------


def _id_map_bytes(chunk_ids: Sequence[str]) -> bytes:
    """Canonical id_map.json bytes (deterministic — sort_keys + fixed
    separators + trailing newline)."""
    doc = {
        "schema_version": _ID_MAP_SCHEMA_VERSION,
        "chunk_ids": list(chunk_ids),
    }
    return (
        json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _manifest_bytes(manifest: VectorIndexManifest) -> bytes:
    """Canonical manifest.json bytes. ``generated_at`` IS written here when
    present, but it is excluded from every content-addressed SHA (the SHAs
    cover embeddings.npy / id_map.json / source chunks, not manifest.json
    itself), so the determinism contract holds modulo this one field."""
    return (
        json.dumps(
            manifest.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _save_embeddings(path: Path, matrix: np.ndarray) -> None:
    """Write embeddings.npy deterministically: float32, C-order, no pickle.

    ``np.save`` writes a fixed header (no timestamps) so identical arrays
    serialize to identical bytes."""
    arr = np.ascontiguousarray(matrix, dtype=np.float32)
    with path.open("wb") as fh:
        np.save(fh, arr, allow_pickle=False)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, _NORM_EPS)
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


# --------------------------------------------------------------------------
# Build API (frozen signature, § 3)
# --------------------------------------------------------------------------


def build_vector_index(
    course_dir: Path,
    *,
    client: Any,  # duck-typed lib.embedding.providers.EmbeddingClient
    chunkset: Optional[str] = None,
    text_field_policy: str = "text+heading",
    force: bool = False,
    generated_at: Optional[str] = None,
) -> VectorIndexManifest:
    """Build (or rebuild) the vector index for a course.

    ``client`` is duck-typed against the frozen ``EmbeddingClient``
    protocol: ``encode_batch(list[str]) -> np.ndarray [n, dim] float32
    L2-normalized``, ``dim`` property, ``model_fingerprint() -> dict``.

    ``force`` is required to overwrite an existing fresh index (one whose
    ``source_chunks_sha256`` still matches the live chunkset) — protects
    against accidental clobbering. A stale or absent index rebuilds
    without ``force``.

    ``generated_at`` is recorded verbatim when provided (callers that want
    determinism leave it ``None``; the determinism contract excludes it).

    Raises ``EmbeddingBackendUnavailable`` (re-raised from the client) when
    the backend can't produce vectors — the build fails closed, it does
    not emit a partial index.
    """
    course_dir = Path(course_dir)
    chunks_path, chunkset_kind = resolve_chunks_for_index(course_dir, chunkset)
    if not chunks_path.exists():
        raise SemanticIndexMissing(
            f"no chunks.jsonl to embed at {chunks_path}; run the chunking "
            f"phase for this course first."
        )

    source_chunks_sha = sha256_file(chunks_path)

    index_dir = course_dir / VECTOR_INDEX_DIRNAME
    manifest_path = index_dir / MANIFEST_FILENAME

    # Refuse to clobber a still-fresh index unless force.
    if manifest_path.exists() and not force:
        try:
            existing = VectorIndexManifest.from_file(manifest_path)
        except Exception:  # noqa: BLE001 — corrupt manifest => allow rebuild.
            existing = None
        if existing is not None and (
            existing.source_chunks_sha256 == source_chunks_sha
        ):
            raise FileExistsError(
                f"a fresh vector index already exists at {index_dir} "
                f"(source_chunks_sha256 matches the live chunkset). Pass "
                f"force=True to rebuild."
            )

    # Asymmetric-retrieval prefixes (D6): a passage prefix (``document_prefix``)
    # is prepended to every embedded passage at build time; the matching query
    # prefix is applied query-side. Read them off the client's resolved
    # provider when present (the registry-backed EmbeddingClient exposes
    # ``resolved.document_prefix`` / ``resolved.query_prefix``); duck-typed
    # fake/test clients without a ``resolved`` attr default to "" (symmetric).
    _resolved = getattr(client, "resolved", None)
    document_prefix = str(getattr(_resolved, "document_prefix", "") or "")
    query_prefix = str(getattr(_resolved, "query_prefix", "") or "")

    chunks = _iter_chunks(chunks_path)
    chunk_ids = [_chunk_id(c) for c in chunks]
    texts = [
        _embed_text(c, text_field_policy, document_prefix) for c in chunks
    ]

    fingerprint = dict(client.model_fingerprint())

    # Graceful-stop (plan D1) — abort-to-paused, first window. An operator
    # ``ed4all stop`` (or a batch-timeout escalation) arms a filesystem
    # sentinel; we raise ``GracefulStopRequested`` here, BEFORE the monolithic
    # encode, so a pending stop never pays for an embedding pass it is about to
    # discard. Embeddings are not LLM calls, so the "≤1 in-flight LLM call"
    # loss guarantee does not bind them — resume re-embeds from scratch. Raising
    # pre-encode (and therefore pre-write) is one half of the "no partial index"
    # contract: no bytes have been written to ``index_dir`` yet.
    check_stop("vector_index_build", 0)

    if texts:
        matrix = np.asarray(client.encode_batch(texts), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(
                f"client.encode_batch returned shape {matrix.shape!r}; "
                f"expected [{len(texts)}, dim]."
            )
        dim = int(matrix.shape[1])
        # Defensive re-normalization: the protocol promises normalized rows,
        # but guaranteeing it here makes search() a pure dot product and
        # keeps the manifest's normalized=true honest.
        matrix = _l2_normalize_rows(matrix)
    else:
        dim = int(fingerprint.get("dim") or getattr(client, "dim", 0) or 0)
        matrix = np.zeros((0, max(dim, 1)), dtype=np.float32)
        if dim <= 0:
            dim = matrix.shape[1]

    # Graceful-stop (plan D1) — abort-to-paused, second window. A stop armed
    # DURING the (potentially long) encode is caught here, still BEFORE any
    # artifact write. This cheap probe is the second half of the "no partial
    # index" contract: the two ``check_stop`` calls bracket the only expensive
    # step, and everything past this point is a short, uninterrupted three-file
    # write sequence. The commit marker (``manifest.json``) is written LAST and
    # carries the SHAs of the already-written ``embeddings.npy`` / ``id_map.json``,
    # so even a non-graceful hard-kill mid-write leaves a manifest-less (or
    # sha-mismatched) dir that the read path fails closed on
    # (``SemanticIndexMissing`` / ``SemanticIndexStale``) — never a half-served
    # partial index. A graceful stop cannot reach the writes at all.
    check_stop("vector_index_build", 0)

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / EMBEDDINGS_FILENAME
    id_map_path = index_dir / ID_MAP_FILENAME

    _save_embeddings(embeddings_path, matrix)
    id_map_path.write_bytes(_id_map_bytes(chunk_ids))

    embeddings_sha = sha256_file(embeddings_path)
    id_map_sha = sha256_text(_id_map_bytes(chunk_ids).decode("utf-8"))

    provider_name = str(fingerprint.get("provider_name") or "")
    kind = str(fingerprint.get("kind") or "")
    model_id = str(fingerprint.get("model_id") or "")
    revision = fingerprint.get("revision")

    # W1b.2 (opt-in ED4ALL_EMBED_OVERFLOW_GUARD, default OFF) — count-and-stamp
    # arm. When the guard is on, record which embedded passages exceed the
    # serving-window token ceiling (ED4ALL_EMBED_MAX_SEQ_TOKENS) so a post-hoc
    # audit can see silent truncation. Pure accounting over the SAME ``texts``
    # that were encoded (document_prefix already prepended); never mutates them.
    # The split arm (ED4ALL_EMBED_OVERFLOW_SPLIT) is deferred — count/stamp only
    # here. Guard off ⇒ embed_overflow stays None ⇒ manifest byte-identical.
    embed_overflow: Optional[Dict[str, Any]] = None
    try:
        from lib.embedding.providers import (
            count_overflow_records,
            resolve_embed_max_seq_tokens,
            resolve_embed_overflow_guard,
        )

        if resolve_embed_overflow_guard():
            max_seq_tokens = resolve_embed_max_seq_tokens()
            overflow_records = [
                {"id": cid, "text": txt}
                for cid, txt in zip(chunk_ids, texts)
            ]
            embed_overflow = count_overflow_records(
                overflow_records, max_seq_tokens
            )
    except Exception:  # noqa: BLE001 — accounting is best-effort, never blocks
        embed_overflow = None

    manifest = VectorIndexManifest(
        schema_version=_MANIFEST_SCHEMA_VERSION,
        embedding_provider=provider_name,
        embedding_kind=kind,
        embedding_model_id=model_id,
        embedding_model_revision=revision if revision is not None else None,
        embedding_dim=dim,
        normalized=True,
        index_type=_INDEX_TYPE,
        chunker_version=str(_CHUNKER_VERSION),
        chunkset_kind=chunkset_kind,
        source_chunks_sha256=source_chunks_sha,
        embeddings_sha256=embeddings_sha,
        id_map_sha256=id_map_sha,
        chunks_count=len(chunk_ids),
        text_field_policy=text_field_policy,
        batch_size=int(fingerprint.get("batch_size") or getattr(client, "batch_size", 0) or 0) or 1,
        device=str(fingerprint.get("device") or getattr(client, "device", "") or "cpu"),
        document_prefix=document_prefix,
        query_prefix=query_prefix,
        embed_overflow=embed_overflow,
        generated_at=generated_at,
    )
    manifest_path.write_bytes(_manifest_bytes(manifest))
    return manifest


# --------------------------------------------------------------------------
# Read API (frozen signature, § 3)
# --------------------------------------------------------------------------


def _resolve_allow_fake(allow_fake: Optional[bool]) -> bool:
    if allow_fake is not None:
        return bool(allow_fake)
    raw = os.environ.get("ED4ALL_EMBEDDING_ALLOW_FAKE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_vector_index(
    course_dir: Path,
    *,
    verify_chunkset: bool = True,
    allow_fake: Optional[bool] = None,
) -> VectorIndex:
    """Load a course's vector index, fail-closed (D7).

    * ``SemanticIndexMissing`` — no ``vector_index/`` dir or missing
      artifacts.
    * ``SemanticIndexStale`` — schema-invalid manifest, on-disk
      sha/count mismatch, or (when ``verify_chunkset``) the live chunkset
      sha no longer matches ``source_chunks_sha256``.
    * ``FakeIndexRefused`` — manifest provider is ``fake`` and fake indexes
      are not allowed (``allow_fake`` is False / env unset).
    """
    course_dir = Path(course_dir)
    index_dir = course_dir / VECTOR_INDEX_DIRNAME
    manifest_path = index_dir / MANIFEST_FILENAME
    embeddings_path = index_dir / EMBEDDINGS_FILENAME
    id_map_path = index_dir / ID_MAP_FILENAME

    if not index_dir.exists() or not manifest_path.exists():
        raise SemanticIndexMissing(
            f"no vector index for {course_dir.name} at {index_dir}; run "
            f"`libv2 vector-index build --course {course_dir.name}`."
        )
    for required in (embeddings_path, id_map_path):
        if not required.exists():
            raise SemanticIndexMissing(
                f"vector index at {index_dir} is incomplete (missing "
                f"{required.name}); rebuild with `libv2 vector-index build "
                f"--course {course_dir.name} --force`."
            )

    manifest = VectorIndexManifest.from_file(manifest_path)

    # Fake-provider production guard (anti-poisoning, D1/D7).
    if manifest.embedding_provider == _FAKE_PROVIDER and not _resolve_allow_fake(
        allow_fake
    ):
        raise FakeIndexRefused(
            f"vector index for {course_dir.name} was built with the 'fake' "
            f"embedding provider; refusing to serve queries. Set "
            f"ED4ALL_EMBEDDING_ALLOW_FAKE=true to override (test-only) or "
            f"rebuild with a real provider."
        )

    # On-disk byte integrity (tamper / partial-write detection).
    actual_emb_sha = sha256_file(embeddings_path)
    if actual_emb_sha != manifest.embeddings_sha256:
        raise SemanticIndexStale(
            f"embeddings.npy sha mismatch for {course_dir.name} "
            f"(manifest {manifest.embeddings_sha256[:16]}... != on-disk "
            f"{actual_emb_sha[:16]}...); rebuild the index."
        )
    actual_id_map_sha = sha256_file(id_map_path)
    if actual_id_map_sha != manifest.id_map_sha256:
        raise SemanticIndexStale(
            f"id_map.json sha mismatch for {course_dir.name}; rebuild the "
            f"index."
        )

    # Live chunkset staleness (the load-bearing fail-closed signal).
    if verify_chunkset:
        chunks_path, _kind = resolve_chunks_for_index(
            course_dir, manifest.chunkset_kind
        )
        if not chunks_path.exists():
            raise SemanticIndexStale(
                f"chunkset {chunks_path} embedded by the index no longer "
                f"exists; rebuild the index."
            )
        live_sha = sha256_file(chunks_path)
        if live_sha != manifest.source_chunks_sha256:
            raise SemanticIndexStale(
                f"vector index for {course_dir.name} is stale: chunks.jsonl "
                f"sha ({live_sha[:16]}...) != manifest source_chunks_sha256 "
                f"({manifest.source_chunks_sha256[:16]}...). Rebuild with "
                f"`libv2 vector-index build --course {course_dir.name} "
                f"--force`."
            )

    # Load + cross-check the arrays.
    with id_map_path.open(encoding="utf-8") as fh:
        id_map_doc = json.load(fh)
    chunk_ids = list(id_map_doc.get("chunk_ids", []))

    matrix = np.load(embeddings_path, allow_pickle=False)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    if matrix.shape[0] != len(chunk_ids):
        raise SemanticIndexStale(
            f"vector index row count ({matrix.shape[0]}) != id_map length "
            f"({len(chunk_ids)}) for {course_dir.name}; rebuild the index."
        )
    if manifest.chunks_count != len(chunk_ids):
        raise SemanticIndexStale(
            f"manifest chunks_count ({manifest.chunks_count}) != id_map "
            f"length ({len(chunk_ids)}) for {course_dir.name}; rebuild."
        )
    if matrix.shape[0] and matrix.shape[1] != manifest.embedding_dim:
        raise SemanticIndexStale(
            f"vector index dim ({matrix.shape[1]}) != manifest "
            f"embedding_dim ({manifest.embedding_dim}) for "
            f"{course_dir.name}; rebuild."
        )

    return VectorIndex(manifest=manifest, chunk_ids=chunk_ids, matrix=matrix)
