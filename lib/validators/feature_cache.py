"""Shared per-phase block-feature cache (``ED4ALL_VALIDATION_FEATURE_CACHE``).

The validation-gate suite (``textbook_to_course::post_rewrite_validation`` = 52
gates) runs SEQUENTIALLY in ``MCP/core/executor.py``: for every gate the input
router re-hydrates all ~424 blocks from ``blocks_final.jsonl``
(``_hydrate_blocks_from_path``) and the ~7 NLI/grounding gates additionally
re-read + re-parse ``chunks.jsonl`` twice. On top of that every heavy validator
re-strips the same HTML, re-splits the same sentences, and re-encodes the same
chunk/objective texts at batch-1 with no cache. The *unique* ML-essential work
is small; the waste is all re-computation.

:class:`BlockFeatureCache` is ONE instance per gate-chain invocation (the
executor builds it immediately before the ``for gate in parsed_gates`` loop and
threads it into every builder + validator). It memoizes, exactly once per phase
run:

CORPUS LAYER (keyed by resolved path + mtime + size):
    * :meth:`blocks` — the hydrated ``List[Block]`` (replaces 52x re-hydration).
    * :meth:`source_chunks` — ``{sourceId -> chunk_text}`` (replaces per-gate
      ``_build_source_chunks_from_dart_jsonl``). Returns a COPY so a builder that
      layers manifest bodies on top (``setdefault``) never corrupts the memo.
    * :meth:`chunk_provenance_index` — ``{provenance_ref -> [chunk_id, ...]}``.
    * :meth:`objectives` — ``(statements, full_dicts)`` flattened once.
    * :meth:`chunk_windows` — ``{sha(chunk_text) -> [window, ...]}`` (the
      ``_sentence_windows`` premises the stage-2 rescue recomputes per block).
    * :meth:`embed` / :meth:`embed_one` — ``{sha(text) -> np.float32[d]}`` via a
      single batched ``encode_batch`` behind one lock (SentenceTransformer is not
      certified for concurrent ``encode``).

PER-BLOCK LAYER (key = ``(block_id, sha256(content))`` — a re-rolled block under
a stable id self-invalidates):
    * :meth:`stripped_text` — ``strip_html_to_text`` (the W-D6 canonical).
    * :meth:`sentences` — keyed BY SPLITTER ID: the three coexisting splitters
      ('claim_support', 'rewrite_grounding') are verdict-bearing and MUST NOT be
      unified.
    * :meth:`resolved_passages` — the block's cited chunks (direct +
      provenance-resolved), mirroring ``_resolve_block_cited_passages``.
    * :meth:`dom_facts` — one HTML parse → a fact bundle (headings, lists,
      tables, links, aria/role/callout/data-cf-* attrs).

NOT cached: ``GroundednessReport``s — the content-addressed JSONL sidecar in
``block_prose_entailment`` already owns cross-run NLI-report persistence. This
cache is in-memory, phase-scoped, never persisted, so there is no second
invalidation story.

Thread-safety: every memo is a dict guarded by a per-feature ``threading.Lock``
with double-checked locking; per-block entries additionally use a per-key
in-flight ``Future`` so N prose-entailment worker threads never duplicate a
strip/split. Embedding access is serialized through a single cache-owned lock.

Default OFF (``ED4ALL_VALIDATION_FEATURE_CACHE`` unset) → the executor never
constructs the cache, every builder/validator sees ``cache=None`` and takes its
legacy self-compute path (byte-identical).
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_FEATURE_CACHE = "ED4ALL_VALIDATION_FEATURE_CACHE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Canonical splitter ids for :meth:`sentences`. Verdict-bearing — do NOT unify.
SPLITTER_CLAIM_SUPPORT = "claim_support"
SPLITTER_REWRITE_GROUNDING = "rewrite_grounding"


def resolve_feature_cache_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_VALIDATION_FEATURE_CACHE`` is a truthy token.

    Parse-with-fallback (mirrors the repo's other opt-in gates): only the
    explicit truthy tokens enable it; anything else (unset / falsey / garbage)
    leaves it OFF so the executor never constructs the cache and every consuming
    seam is byte-identical.
    """
    src = env if env is not None else os.environ
    return str(src.get(ENV_FEATURE_CACHE, "")).strip().lower() in _TRUTHY


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _path_key(path: Path) -> str:
    """Resolved-path + mtime + size key so a rewritten file self-invalidates."""
    try:
        st = path.stat()
        return f"{path.resolve()}::{st.st_mtime_ns}::{st.st_size}"
    except OSError:
        return f"{path}::missing"


class _MemoDict:
    """A lock-guarded memo with double-checked locking + per-key in-flight Future.

    ``get_or_compute(key, fn)`` runs ``fn`` exactly once per key even under N
    concurrent callers: the first caller installs a ``Future`` and computes; late
    callers block on that ``Future`` instead of duplicating the work. A compute
    exception is surfaced to every waiter and the key is cleared so a later call
    can retry.
    """

    __slots__ = ("_store", "_pending", "_lock")

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._pending: Dict[str, "Future"] = {}
        self._lock = threading.Lock()

    def get_or_compute(self, key: str, fn) -> Any:
        with self._lock:
            if key in self._store:
                return self._store[key]
            fut = self._pending.get(key)
            if fut is None:
                fut = Future()
                self._pending[key] = fut
                owner = True
            else:
                owner = False
        if owner:
            try:
                value = fn()
            except BaseException as exc:  # noqa: BLE001 — surface to every waiter
                with self._lock:
                    self._pending.pop(key, None)
                fut.set_exception(exc)
                raise
            with self._lock:
                self._store[key] = value
                self._pending.pop(key, None)
            fut.set_result(value)
            return value
        return fut.result()


class BlockFeatureCache:
    """Compute-once, phase-scoped, thread-safe feature cache shared across gates.

    Constructed by the executor from ``(phase_outputs, workflow_params)`` right
    before the gate loop; passed to every builder (``cache=``) and validator
    (``inputs["feature_cache"]``). Every consumer falls back to self-compute when
    the cache is absent, so gates stay independently testable and the default
    (flag-off) path is byte-identical.
    """

    def __init__(
        self,
        phase_outputs: Optional[Dict[str, Any]] = None,
        workflow_params: Optional[Dict[str, Any]] = None,
        *,
        embedder: Any = None,
        embedder_model_name: Optional[str] = None,
    ) -> None:
        self.phase_outputs = phase_outputs or {}
        self.workflow_params = workflow_params or {}

        # Corpus-layer memos (keyed by path/mtime/size or content sha).
        self._blocks_memo = _MemoDict()
        self._source_chunks_memo = _MemoDict()
        self._provenance_memo = _MemoDict()
        self._objectives_memo = _MemoDict()
        self._chunk_windows = _MemoDict()

        # Per-block-layer memos.
        self._stripped = _MemoDict()
        self._sentences = _MemoDict()
        self._passages = _MemoDict()
        self._dom_facts = _MemoDict()

        # Embedding memo — a single dict guarded by one lock; encode_batch runs
        # under that lock (SentenceTransformer is not concurrency-certified).
        self._embed_store: Dict[str, Any] = {}
        self._embed_lock = threading.Lock()
        self._embedder = embedder
        self._embedder_probed = embedder is not None
        self._embedder_model_name = embedder_model_name

    # ---------------------------------------------------------------- keys --- #

    @staticmethod
    def block_key(block: Any) -> str:
        """``(block_id, sha256(content))`` content-addressed per-block key.

        Content-sha because ``courseforge-rewrite`` re-rolls change a block's
        content under a stable ``block_id``; hashing the content means a
        re-rolled block self-invalidates in-phase.
        """
        block_id = str(getattr(block, "block_id", "") or "")
        content = getattr(block, "content", "")
        content_str = content if isinstance(content, str) else repr(content)
        return f"{block_id}::{_sha(content_str)}"

    # ------------------------------------------------------------- corpus --- #

    def blocks(self, blocks_path: Any) -> List[Any]:
        """Hydrated ``List[Block]`` for ``blocks_path`` (memoized, path+mtime+size).

        Replaces the 52x ``_hydrate_blocks_from_path`` re-hydration. Returns the
        SAME list across gates (treated read-only by every validator).
        """
        path = Path(blocks_path)
        key = _path_key(path)

        def _compute() -> List[Any]:
            from MCP.hardening.gate_input_routing import _hydrate_blocks_from_path

            return _hydrate_blocks_from_path(path)

        return self._blocks_memo.get_or_compute(key, _compute)

    def source_chunks(
        self, phase_outputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """``{sourceId -> chunk_text}`` from the DART chunkset (memoized).

        Returns a shallow COPY so a builder that layers manifest bodies on top
        via ``setdefault`` never mutates the shared memo (the JSONL parse — the
        expensive part — is what is cached).
        """
        po = phase_outputs if phase_outputs is not None else self.phase_outputs

        def _compute() -> Dict[str, str]:
            from MCP.hardening.gate_input_routing import (
                _build_source_chunks_from_dart_jsonl,
            )

            return _build_source_chunks_from_dart_jsonl(po)

        memo = self._source_chunks_memo.get_or_compute("__default__", _compute)
        return dict(memo)

    def chunk_provenance_index(
        self, phase_outputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """``{provenance_ref -> [chunk_id, ...]}`` reverse index (memoized)."""
        po = phase_outputs if phase_outputs is not None else self.phase_outputs

        def _compute() -> Dict[str, List[str]]:
            from MCP.hardening.gate_input_routing import (
                _build_chunk_provenance_index_from_dart_jsonl,
            )

            return _build_chunk_provenance_index_from_dart_jsonl(po)

        return self._provenance_memo.get_or_compute("__default__", _compute)

    def objectives(
        self, objectives_path: Any
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """``(objective_statements, objectives_full_dicts)`` flattened once.

        Mirrors the ``_build_block_statistical_input`` flatten so every
        statistical-tier validator sees the same maps without re-loading +
        re-flattening ``synthesized_objectives.json`` per gate. Returns fresh top
        dicts (values shared) so a consumer can't inject keys into the memo.
        """
        path = Path(objectives_path)
        key = _path_key(path)

        def _compute() -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
            import json as _json

            from lib.validators.abcd_objective import _flatten_objectives

            statements: Dict[str, str] = {}
            full_dicts: Dict[str, Dict[str, Any]] = {}
            if not path.exists():
                return statements, full_dicts
            try:
                payload = _json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.debug("feature_cache.objectives read %s failed: %s", path, exc)
                return statements, full_dicts
            for lo in _flatten_objectives(payload):
                if not isinstance(lo, dict):
                    continue
                lo_id_raw = lo.get("id") or lo.get("objective_id")
                if not isinstance(lo_id_raw, str) or not lo_id_raw:
                    continue
                statement_raw = lo.get("statement") or lo.get("text")
                if isinstance(statement_raw, str) and statement_raw.strip():
                    statements[lo_id_raw] = statement_raw.strip()
                full_dicts[lo_id_raw] = dict(lo)
            return statements, full_dicts

        statements, full_dicts = self._objectives_memo.get_or_compute(key, _compute)
        return dict(statements), dict(full_dicts)

    def chunk_windows(self, chunk_text: str) -> List[str]:
        """``_sentence_windows(chunk_text)`` memoized by content sha.

        The stage-2 windowed-rescue premises the grounded-answer scorer
        recomputes per rescuing block. Cached once per distinct chunk text.
        """
        key = _sha(chunk_text)

        def _compute() -> List[str]:
            from lib.retrieval.groundedness import _sentence_windows

            return _sentence_windows(chunk_text)

        return self._chunk_windows.get_or_compute(key, _compute)

    # --------------------------------------------------------- per-block --- #

    def stripped_text(self, block: Any) -> str:
        """``strip_html_to_text(block.content)`` (the W-D6 canonical), memoized."""
        key = self.block_key(block)

        def _compute() -> str:
            from lib.utils import strip_html_to_text

            content = getattr(block, "content", "")
            if not isinstance(content, str):
                return ""
            return strip_html_to_text(content)

        return self._stripped.get_or_compute(key, _compute)

    def sentences(self, block: Any, splitter_id: str) -> List[str]:
        """Sentence list for ``block`` keyed BY ``splitter_id``.

        The three coexisting splitters are verdict-bearing; keying by id keeps
        them distinct so grounded_sentence_rate / entailment denominators never
        silently change. Supported ids: ``'claim_support'`` (groundedness
        ``_split_claims_for_scoring`` — returns the kept claims only) and
        ``'rewrite_grounding'`` (rewrite_source_grounding ``_segment_sentences``
        filtered by ``_is_non_trivial``).
        """
        key = f"{splitter_id}::{self.block_key(block)}"

        def _compute() -> List[str]:
            text = self.stripped_text(block)
            if splitter_id == SPLITTER_CLAIM_SUPPORT:
                from lib.retrieval.groundedness import _split_claims_for_scoring

                kept, _filtered = _split_claims_for_scoring(text)
                return kept
            if splitter_id == SPLITTER_REWRITE_GROUNDING:
                from lib.validators.rewrite_source_grounding import (
                    _is_non_trivial,
                    _segment_sentences,
                )

                return [s for s in _segment_sentences(text) if _is_non_trivial(s)]
            raise ValueError(f"unknown splitter_id: {splitter_id!r}")

        return self._sentences.get_or_compute(key, _compute)

    def resolved_passages(
        self,
        block: Any,
        source_chunks: Dict[str, str],
        provenance_index: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, str]]:
        """The block's cited chunks (direct + provenance-resolved), memoized.

        Mirrors ``block_prose_entailment._resolve_block_cited_passages`` (incl.
        the ADD-only provenance pass). Keyed on the block content-sha AND the
        provenance-resolution mode so a flag flip never serves a stale passage
        set.
        """
        prov_flag = "1" if provenance_index else "0"
        key = f"{prov_flag}::{self.block_key(block)}"

        def _compute() -> List[Dict[str, str]]:
            from lib.validators.block_prose_entailment import (
                _resolve_block_cited_passages,
            )

            return _resolve_block_cited_passages(
                block, source_chunks, provenance_index
            )

        return self._passages.get_or_compute(key, _compute)

    def dom_facts(self, block: Any) -> Dict[str, Any]:
        """One HTML parse per block → a structural fact bundle (best-effort).

        Consumed opt-in by the DOM gates (rewrite_html_shape / interactive_a11y /
        callout_structure / ...). Returns ``{}`` when the content is non-string
        or the HTML parser is unavailable, so a consumer degrades to its own
        parse exactly as without the cache.
        """
        key = self.block_key(block)

        def _compute() -> Dict[str, Any]:
            content = getattr(block, "content", "")
            if not isinstance(content, str) or not content.strip():
                return {}
            try:
                from bs4 import BeautifulSoup  # type: ignore
            except Exception:  # noqa: BLE001 — no parser → let consumer self-compute
                return {}
            try:
                soup = BeautifulSoup(content, "html.parser")
            except Exception as exc:  # noqa: BLE001
                logger.debug("feature_cache.dom_facts parse failed: %s", exc)
                return {}
            headings = [
                h.name for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            ]
            links = [
                a.get("href", "") for a in soup.find_all("a") if a.get("href")
            ]
            roles = [
                el.get("role", "") for el in soup.find_all(attrs={"role": True})
            ]
            aria = sorted(
                {
                    attr
                    for el in soup.find_all(True)
                    for attr in getattr(el, "attrs", {})
                    if attr.startswith("aria-")
                }
            )
            data_cf = sorted(
                {
                    attr
                    for el in soup.find_all(True)
                    for attr in getattr(el, "attrs", {})
                    if attr.startswith("data-cf-")
                }
            )
            callout_classes = sorted(
                {
                    cls
                    for el in soup.find_all(class_=True)
                    for cls in (el.get("class") or [])
                    if "callout" in cls.lower()
                }
            )
            return {
                "heading_tags": headings,
                "heading_count": len(headings),
                "list_count": len(soup.find_all(["ul", "ol"])),
                "list_item_count": len(soup.find_all("li")),
                "table_count": len(soup.find_all("table")),
                "link_count": len(links),
                "links": links,
                "img_count": len(soup.find_all("img")),
                "roles": roles,
                "aria_attrs": aria,
                "data_cf_attrs": data_cf,
                "callout_classes": callout_classes,
            }

        return self._dom_facts.get_or_compute(key, _compute)

    # -------------------------------------------------------- embeddings --- #

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        """Return ``{sha(text) -> vector}`` for ``texts``, batching the misses.

        Collects the cache-miss texts, calls ``SentenceEmbedder.encode_batch``
        ONCE under the cache-owned lock (SentenceTransformer is not
        concurrency-certified), memoizes each vector by its content sha, and
        returns the full ``{sha -> vector}`` map. Returns an empty map when no
        embedder is available (caller degrades exactly as today).
        """
        out: Dict[str, Any] = {}
        misses: List[str] = []
        miss_shas: List[str] = []
        seen_miss: set = set()
        with self._embed_lock:
            for text in texts:
                h = _sha(text)
                cached = self._embed_store.get(h)
                if cached is not None:
                    out[h] = cached
                elif h not in seen_miss:
                    seen_miss.add(h)
                    misses.append(text)
                    miss_shas.append(h)
            if misses:
                embedder = self._resolve_embedder_locked()
                if embedder is not None:
                    try:
                        vectors = self._encode_misses(embedder, misses)
                    except Exception as exc:  # noqa: BLE001 — degrade, no fabrication
                        logger.debug("feature_cache encode_batch failed: %s", exc)
                        vectors = None
                    if vectors is not None:
                        for h, vec in zip(miss_shas, vectors):
                            self._embed_store[h] = vec
                            out[h] = vec
        return out

    @staticmethod
    def _encode_misses(embedder: Any, misses: List[str]) -> Any:
        """Embed the miss texts, honouring the Builder-C incremental-embed flags.

        Default (both flags off) → ``embedder.encode_batch(misses)`` — the exact
        legacy call, byte-identical. ``ED4ALL_EMBED_BATCH_TUNE`` adds
        ``batch_size``/``length_sort``; ``ED4ALL_EMBED_PERSIST_CACHE`` routes
        through the disk-persisted ``encode_batch_cached`` (falls back to
        ``encode_batch`` when the embedder predates that method).
        """
        from lib.embedding.sentence_embedder import (
            _TUNE_BATCH_SIZE,
            resolve_embed_batch_tune,
            resolve_embed_persist_cache,
        )

        kwargs: Dict[str, Any] = {}
        if resolve_embed_batch_tune():
            kwargs = {"batch_size": _TUNE_BATCH_SIZE, "length_sort": True}
        if resolve_embed_persist_cache():
            cached = getattr(embedder, "encode_batch_cached", None)
            if callable(cached):
                return cached(misses, **kwargs)
        return embedder.encode_batch(misses, **kwargs)

    def _resolve_embedder_locked(self) -> Any:
        """Embedder resolve assuming ``self._embed_lock`` is already held."""
        if self._embedder is not None or self._embedder_probed:
            return self._embedder
        try:
            from lib.embedding.sentence_embedder import try_load_embedder

            if self._embedder_model_name:
                self._embedder = try_load_embedder(self._embedder_model_name)
            else:
                self._embedder = try_load_embedder()
        except Exception as exc:  # noqa: BLE001 — degrade to no embedder
            logger.debug("feature_cache embedder load failed: %s", exc)
            self._embedder = None
        self._embedder_probed = True
        return self._embedder

    def embed_one(self, text: str) -> Any:
        """Single-text convenience over :meth:`embed`; ``None`` when unavailable."""
        return self.embed([text]).get(_sha(text))


__all__ = [
    "ENV_FEATURE_CACHE",
    "SPLITTER_CLAIM_SUPPORT",
    "SPLITTER_REWRITE_GROUNDING",
    "BlockFeatureCache",
    "resolve_feature_cache_enabled",
]
