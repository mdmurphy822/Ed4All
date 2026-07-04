"""Cross-encoder reranker over the first-stage retrieval candidate pool.

Registry-entry pattern (NOT subclasses), mirroring
``lib/embedding/providers.py``: a new reranker backend is a registry-entry
change to :data:`_RERANKER_PROVIDERS`, never a new subclass.

ADR-002 keeps cross-encoder rerankers OUT of the LibV2 *reference* retrieval
surface ("inference cost, model-version churn, hyperparameter surface ...
belong downstream"). This module is the ADR-compliant downstream home: it lives
in ``lib/retrieval/`` (the consumer of LibV2 retrieval) and is wired ONLY into
the grounded-answer path, strictly between first-stage retrieve and the refusal
gate.

Two execution kinds:

- ``"cross-encoder"`` — in-process ``sentence-transformers`` ``CrossEncoder``
  (the ``[embedding]`` pyproject extra already ships it; NO new dependency).
  ``offline``-friendly: a missing-weights / absent-library load raises
  :class:`RerankerBackendUnavailable`. The default model is the license-clean
  ``BAAI/bge-reranker-base`` (MIT). See ``docs/LICENSING.md`` § "Reranker
  providers".
- ``"fake"`` — deterministic test provider (``sha256(query+text)`` → seeded
  score). A REAL registry entry so the reorder/trim/anti-fabrication paths run
  in CI with zero weights and zero network. Production-poisoning is blocked: the
  grounded-answer hook refuses ``provider="fake"`` unless
  ``ED4ALL_RERANK_ALLOW_FAKE`` is set (mirrors ``ED4ALL_EMBEDDING_ALLOW_FAKE`` /
  the ``LOCAL_DISPATCHER_ALLOW_STUB`` precedent).

DEFAULT OFF: ``ED4ALL_RERANK_PROVIDER`` unset → :func:`reranker_configured`
returns ``None`` and the whole module is a strict no-op (no client built, no
over-fetch, retrieved order + native scores byte-identical). Garbage / unknown
provider name → off (parse-with-fallback, fail-OPEN in the hook).

Anti-fabrication contract: rerank is REORDER + TRIM only. The kept set is a
strict subset of the retrieved set, always-keep-≥1, and the passages' NATIVE
first-stage scores are preserved verbatim (only the LIST ORDER changes) so the
per-engine/per-embedding-model refusal threshold stays calibrated. The
cross-encoder score is recorded for the decision capture only — never written
back onto the passage.
"""
from __future__ import annotations

import hashlib
import logging
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Truthy-env helper — same convention as lib/embedding/providers.py.
# ---------------------------------------------------------------------------
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_VALUES


# ---------------------------------------------------------------------------
# Env-var names (single source of truth for the ED4ALL_RERANK_* family).
# Documented in docs/operations/behavior-flags.md (indexed in root CLAUDE.md).
# ---------------------------------------------------------------------------
ENV_PROVIDER = "ED4ALL_RERANK_PROVIDER"
ENV_MODEL = "ED4ALL_RERANK_MODEL"
ENV_CANDIDATE_POOL = "ED4ALL_RERANK_CANDIDATE_POOL"
ENV_DEVICE = "ED4ALL_RERANK_DEVICE"
ENV_BATCH_SIZE = "ED4ALL_RERANK_BATCH_SIZE"
ENV_ALLOW_FAKE = "ED4ALL_RERANK_ALLOW_FAKE"

DEFAULT_CANDIDATE_POOL = 30
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 16

# Decision-capture enum member (registered in
# schemas/events/decision_event.schema.json in the same edit as this emitter).
DECISION_TYPE_RERANK = "grounded_answer_rerank"


# ---------------------------------------------------------------------------
# Provider registry — registry entries, NOT subclasses (operator standing rule).
# ---------------------------------------------------------------------------
_RERANKER_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "st-cross-encoder": {
        "kind": "cross-encoder",
        "model_env": ENV_MODEL,
        "model_default": "BAAI/bge-reranker-base",  # MIT, license-clean default
        "device_env": ENV_DEVICE,
        "device_default": DEFAULT_DEVICE,
        "batch_size_env": ENV_BATCH_SIZE,
        "batch_size_default": DEFAULT_BATCH_SIZE,
    },
    "fake": {
        "kind": "fake",
        "model_default": "fake-deterministic-v1",
        "batch_size_default": DEFAULT_BATCH_SIZE,
    },
}


# ---------------------------------------------------------------------------
# Candidate model registry with license metadata. All MIT / Apache-2.0.
# jina / mxbai rerankers are deliberately EXCLUDED (CC-BY-NC / non-clean).
# ---------------------------------------------------------------------------
RERANKER_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BAAI/bge-reranker-base": {
        "license": "MIT",
        "trust_remote_code": False,
        "notes": "default — license-clean cross-encoder; in-process CrossEncoder",
    },
    "BAAI/bge-reranker-large": {
        "license": "MIT",
        "trust_remote_code": False,
        "notes": "benchmark candidate — larger bge cross-encoder",
    },
    "BAAI/bge-reranker-v2-m3": {
        "license": "Apache-2.0",
        "trust_remote_code": False,
        "notes": "benchmark candidate — multilingual headroom",
    },
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RerankerBackendUnavailable(RuntimeError):
    """Weights/library absent or refused. The grounded-answer hook catches
    this and FAILS OPEN (original first-stage order), never crashing or
    downgrading grounding. Raised when:

    - ``sentence-transformers`` is not installed for a ``cross-encoder`` provider;
    - a ``cross-encoder`` model load / predict fails for any reason;
    - a ``fake``-provider reranker is used on a production read path without
      ``ED4ALL_RERANK_ALLOW_FAKE`` (anti-poisoning).
    """


# ---------------------------------------------------------------------------
# Resolved provider (frozen contract)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedReranker:
    """A fully-resolved reranker provider (post env-var resolution)."""

    provider_name: str
    kind: str  # "cross-encoder" | "fake"
    model_id: str
    device: Optional[str]
    batch_size: int
    trust_remote_code: bool = False
    license: Optional[str] = None


def reranker_configured() -> Optional[str]:
    """Return the configured provider name from ``ED4ALL_RERANK_PROVIDER``.

    Returns ``None`` when the env var is unset / empty / whitespace — the
    strict-no-op signal the grounded-answer hook reads to keep the off path
    byte-identical (no over-fetch, no client built). An UNKNOWN (but non-empty)
    name is returned verbatim so the caller can fail OPEN on resolution.
    """
    raw = os.environ.get(ENV_PROVIDER, "").strip()
    return raw or None


def resolve_candidate_pool(candidate_pool: Optional[int] = None) -> int:
    """Resolve the over-fetch candidate-pool size.

    Resolution chain: explicit arg → ``ED4ALL_RERANK_CANDIDATE_POOL`` →
    :data:`DEFAULT_CANDIDATE_POOL`. Garbage / non-positive → default
    (parse-with-fallback, mirroring the embedding batch-size resolver).
    """
    if candidate_pool is not None and candidate_pool > 0:
        return int(candidate_pool)
    raw = os.environ.get(ENV_CANDIDATE_POOL)
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
            logger.warning(
                "%s=%r is not a positive int; using default %d",
                ENV_CANDIDATE_POOL, raw, DEFAULT_CANDIDATE_POOL,
            )
        except ValueError:
            logger.warning(
                "%s=%r is not an int; using default %d",
                ENV_CANDIDATE_POOL, raw, DEFAULT_CANDIDATE_POOL,
            )
    return DEFAULT_CANDIDATE_POOL


def allow_fake_enabled() -> bool:
    """Return True when ``ED4ALL_RERANK_ALLOW_FAKE`` is truthy.

    The grounded-answer hook calls this to decide whether a ``fake``-provider
    reranker may run on a production read path. Mirrors
    ``ED4ALL_EMBEDDING_ALLOW_FAKE`` / ``LOCAL_DISPATCHER_ALLOW_STUB``.
    """
    return _env_truthy(ENV_ALLOW_FAKE)


def resolve_reranker_provider(
    provider_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> ResolvedReranker:
    """Resolve a registry entry into a :class:`ResolvedReranker`.

    Resolution chain mirrors :func:`resolve_embedding_provider`: explicit arg →
    env var → registry default. ``provider_name`` defaults to
    ``ED4ALL_RERANK_PROVIDER``.

    Unknown ``provider_name`` raises ``ValueError`` listing every registered
    provider — add an entry to :data:`_RERANKER_PROVIDERS`, never a subclass.
    The grounded-answer hook catches that ValueError and fails OPEN; this
    low-level resolver still raises so the contract is explicit + testable.
    """
    resolved_provider = (
        provider_name
        or os.environ.get(ENV_PROVIDER)
        or ""
    ).strip()
    entry = _RERANKER_PROVIDERS.get(resolved_provider)
    if entry is None:
        registered = sorted(_RERANKER_PROVIDERS.keys())
        raise ValueError(
            f"Unknown reranker provider: {resolved_provider!r}. "
            f"Registered providers: {registered}. Add a registry entry to "
            f"_RERANKER_PROVIDERS in lib/retrieval/reranker.py — no "
            f"subclassing required."
        )

    kind = entry["kind"]

    model_env = entry.get("model_env")
    resolved_model = (
        model_id
        or (os.environ.get(model_env) if model_env else None)
        or entry.get("model_default")
    )
    if not resolved_model:
        raise ValueError(
            f"reranker provider {resolved_provider!r} resolved to an empty "
            f"model_id; set {model_env or ENV_MODEL} or pass model_id="
        )

    # Batch size: env → registry default (parse-with-fallback).
    batch_env = entry.get("batch_size_env")
    batch_default = int(entry.get("batch_size_default", DEFAULT_BATCH_SIZE))
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
    if kind == "cross-encoder":
        device_env = entry.get("device_env")
        device = (
            (os.environ.get(device_env) if device_env else None)
            or entry.get("device_default")
            or DEFAULT_DEVICE
        )

    model_meta = RERANKER_MODEL_REGISTRY.get(resolved_model, {})
    trust_remote_code = bool(model_meta.get("trust_remote_code", False))
    license_id = model_meta.get("license")

    return ResolvedReranker(
        provider_name=resolved_provider,
        kind=kind,
        model_id=resolved_model,
        device=device,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        license=license_id,
    )


# ---------------------------------------------------------------------------
# Fake deterministic scorer
# ---------------------------------------------------------------------------
def _fake_score(query: str, text: str) -> float:
    """Deterministic relevance score for ``(query, text)``.

    ``sha256(query \x00 text)`` → leading 8 bytes → uint64 → [0, 1). Same
    inputs → byte-identical score across processes / machines (pure stdlib).
    """
    digest = hashlib.sha256(f"{query}\x00{text}".encode("utf-8")).digest()
    (val,) = struct.unpack(">Q", digest[:8])
    return val / float(1 << 64)


# ---------------------------------------------------------------------------
# Reranker client — ONE class, kind-dispatched internally.
# ---------------------------------------------------------------------------
class RerankerClient:
    """Score ``(query, passage_text)`` pairs → relevance floats. Kind-dispatched.

    ONE class, NO per-kind subclasses — the ``kind`` field on the resolved
    provider drives an internal dispatch. ``score`` is the single hot path.
    A higher score means MORE relevant. Any backend failure raises
    :class:`RerankerBackendUnavailable`.
    """

    def __init__(
        self,
        resolved: ResolvedReranker,
        *,
        model: Optional[Any] = None,
    ) -> None:
        self.resolved = resolved
        self._model = model

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        kind = self.resolved.kind
        if kind == "fake":
            return [_fake_score(query, t) for t in texts]
        if kind == "cross-encoder":
            return self._score_cross_encoder(query, list(texts))
        raise RerankerBackendUnavailable(
            f"unsupported reranker kind: {kind!r}"
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        r = self.resolved
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RerankerBackendUnavailable(
                "sentence-transformers is not installed; install the "
                "[embedding] extra (`pip install -e .[embedding]`) or unset "
                f"{ENV_PROVIDER}. Underlying error: {exc}"
            ) from exc
        kwargs: Dict[str, Any] = {"device": r.device or DEFAULT_DEVICE}
        if r.trust_remote_code:
            kwargs["trust_remote_code"] = True
        try:
            self._model = CrossEncoder(r.model_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 — missing cache / load failure
            raise RerankerBackendUnavailable(
                f"failed to load CrossEncoder model {r.model_id!r} "
                f"(device={r.device}): {exc}"
            ) from exc
        return self._model

    def _score_cross_encoder(self, query: str, texts: List[str]) -> List[float]:
        model = self._ensure_model()
        pairs = [(query, t) for t in texts]
        try:
            scores = model.predict(
                pairs,
                batch_size=self.resolved.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 — runtime model failure
            raise RerankerBackendUnavailable(
                f"CrossEncoder predict failed for model "
                f"{self.resolved.model_id!r}: {exc}"
            ) from exc
        return [float(s) for s in scores]


def build_reranker_client(
    resolved: Optional[ResolvedReranker] = None,
    *,
    provider_name: Optional[str] = None,
    model_id: Optional[str] = None,
    model: Optional[Any] = None,
) -> RerankerClient:
    """Resolve (if needed) + construct a :class:`RerankerClient`."""
    if resolved is None:
        resolved = resolve_reranker_provider(provider_name, model_id)
    return RerankerClient(resolved, model=model)


# ---------------------------------------------------------------------------
# Reorder + trim (anti-fabrication: subset, always-keep-≥1, native score kept)
# ---------------------------------------------------------------------------
def rerank_retrieved_passages(
    query: str,
    passages: Sequence[Any],
    *,
    top_k: int,
    client: RerankerClient,
    resolved: Optional[ResolvedReranker] = None,
    capture: Optional[Any] = None,
    course_slug: str = "",
    query_sha: str = "",
) -> List[Any]:
    """Re-score, REORDER, and TRIM ``passages`` to ``top_k``.

    Operates on any objects exposing ``.text`` / ``.chunk_id`` / ``.score``
    (e.g. :class:`~lib.retrieval.answer_composer.RetrievedPassage`). The
    passage objects are NEVER mutated — only the list order changes — so each
    passage's native first-stage ``.score`` is preserved verbatim (the
    refusal threshold stays calibrated). The cross-encoder score is used only
    for the sort + the decision-capture rationale.

    Anti-fabrication: the returned list is a strict subset of ``passages``
    (a stable re-ordering then a head-trim to ``top_k``); no id is invented;
    at least one passage is always kept when the input is non-empty.
    """
    passages = list(passages)
    if not passages:
        return passages
    resolved = resolved or client.resolved

    texts = [str(getattr(p, "text", "") or "") for p in passages]
    rerank_scores = client.score(query, texts)
    if len(rerank_scores) != len(passages):
        # Honest mismatch → fail open (caller logs); never fabricate.
        raise RerankerBackendUnavailable(
            f"reranker returned {len(rerank_scores)} scores for "
            f"{len(passages)} passages"
        )

    original_ids = [str(getattr(p, "chunk_id", "")) for p in passages]
    # Stable descending sort by rerank score; ties keep first-stage order.
    order = sorted(
        range(len(passages)),
        key=lambda i: (-rerank_scores[i], i),
    )
    reordered = [passages[i] for i in order]

    effective_top_k = max(1, min(int(top_k), len(reordered)))
    kept = reordered[:effective_top_k]

    _emit_rerank_capture(
        capture,
        resolved=resolved,
        course_slug=course_slug,
        query_sha=query_sha,
        original_ids=original_ids,
        kept_ids=[str(getattr(p, "chunk_id", "")) for p in kept],
        rerank_scores=rerank_scores,
        order=order,
        top_k=effective_top_k,
        fallback_used=False,
    )
    return kept


def _emit_rerank_capture(
    capture: Optional[Any],
    *,
    resolved: ResolvedReranker,
    course_slug: str,
    query_sha: str,
    original_ids: Sequence[str],
    kept_ids: Sequence[str],
    rerank_scores: Sequence[float],
    order: Sequence[int],
    top_k: int,
    fallback_used: bool,
) -> None:
    if capture is None:
        return
    pool = len(original_ids)
    # Rank churn: positions whose id changed between first-stage and reranked.
    reordered_ids = [original_ids[i] for i in order]
    rank_churn = sum(
        1 for a, b in zip(original_ids, reordered_ids) if a != b
    )
    top1_changed = bool(original_ids and reordered_ids and original_ids[0] != reordered_ids[0])
    score_max = max(rerank_scores) if rerank_scores else 0.0
    score_min = min(rerank_scores) if rerank_scores else 0.0
    rationale = (
        f"reranked query {query_sha or '?'} (course={course_slug or '?'}) "
        f"with provider={resolved.provider_name} model={resolved.model_id} "
        f"candidate_pool={pool} top_k={top_k} kept={len(kept_ids)}; "
        f"rank_churn={rank_churn} top1_changed={top1_changed} "
        f"score_max={score_max:.4f} score_min={score_min:.4f} "
        f"fallback_used={fallback_used}; native first-stage scores preserved "
        f"(reorder+trim only, kept set is a subset of the retrieved set)"
    )
    try:
        capture.log_decision(
            decision_type=DECISION_TYPE_RERANK,
            decision=f"rerank:{resolved.provider_name}",
            rationale=rationale,
            context=f"top_k={top_k} pool={pool} churn={rank_churn}",
        )
    except Exception:  # pragma: no cover — capture must never break the path
        pass


def maybe_rerank(
    query: str,
    passages: Sequence[Any],
    *,
    top_k: int,
    capture: Optional[Any] = None,
    course_slug: str = "",
    query_sha: str = "",
    client: Optional[RerankerClient] = None,
) -> List[Any]:
    """Env-gated, FAIL-OPEN reranker entry point for the grounded-answer hook.

    Reads ``ED4ALL_RERANK_PROVIDER``; when unset/empty returns ``passages``
    unchanged (the byte-identical-when-off contract — the caller must also
    avoid over-fetching in that case). When configured, resolves + builds the
    client and reorders+trims to ``top_k``. ANY failure (unknown provider,
    missing weights, sentence-transformers absent, fake-without-allow,
    score-count mismatch) logs a warning and returns the ORIGINAL first-stage
    order — never raises, never downgrades grounding.
    """
    provider_name = reranker_configured()
    if provider_name is None:
        return list(passages)
    try:
        resolved = resolve_reranker_provider(provider_name)
        if resolved.kind == "fake" and not allow_fake_enabled():
            raise RerankerBackendUnavailable(
                "fake-provider reranker refused on a production read path; "
                f"set {ENV_ALLOW_FAKE}=1 to permit it (anti-poisoning)."
            )
        if client is None:
            client = build_reranker_client(resolved)
        return rerank_retrieved_passages(
            query,
            passages,
            top_k=top_k,
            client=client,
            resolved=resolved,
            capture=capture,
            course_slug=course_slug,
            query_sha=query_sha,
        )
    except Exception as exc:  # noqa: BLE001 — fail OPEN
        logger.warning(
            "reranker disabled (fail-open) for provider %r: %s",
            provider_name, exc,
        )
        return list(passages)


__all__ = [
    "RerankerBackendUnavailable",
    "ResolvedReranker",
    "RerankerClient",
    "RERANKER_MODEL_REGISTRY",
    "DECISION_TYPE_RERANK",
    "ENV_PROVIDER",
    "ENV_MODEL",
    "ENV_CANDIDATE_POOL",
    "ENV_DEVICE",
    "ENV_BATCH_SIZE",
    "ENV_ALLOW_FAKE",
    "DEFAULT_CANDIDATE_POOL",
    "reranker_configured",
    "resolve_reranker_provider",
    "resolve_candidate_pool",
    "allow_fake_enabled",
    "build_reranker_client",
    "rerank_retrieved_passages",
    "maybe_rerank",
]
