"""Retrieval vertical service — REAL wrappers, no stubs.

Wraps the production LibV2 retriever (BM25 + multi-query), trained-adapter
inference (PEFT/LoRA), and a live LLM-driven rerank against the active
settings provider/model. There are NO fabricated generations or fake
rankings anywhere in this module: every path either calls the real backend
or returns a typed error dict the router maps to an HTTP status.

Import contract (spec §0 / §5):
- This module imports cleanly WITHOUT FastAPI, torch, transformers, peft, or
  the LibV2 ``tools`` package. Every heavy / LibV2 import happens lazily
  inside the function that needs it.
- ``LibV2/tools/`` is NOT a Python package (no ``__init__.py``), so
  ``tools.libv2.*`` is only importable with ``LibV2/`` on ``sys.path``. The
  LibV2 CLI is run as ``python -m tools.libv2.cli`` from inside ``LibV2/``;
  to make the same modules importable from the repo root we prepend
  ``lib.paths.LIBV2_PATH`` (honoring ``ED4ALL_LIBV2_ROOT``) to ``sys.path``
  inside ``_libv2_root_on_path`` before importing ``tools.libv2.*``.

Adapter inference licensing note: AdapterCallable loads the trained adapter
locally and runs it — this is inference of an already-trained model, not
training-data synthesis, so the LICENSING posture for synthesis providers
does not apply here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Top-of-module cap on top_k (spec §5: "Honor top_k cap 50").
TOP_K_CAP = 50


# --------------------------------------------------------------------- paths


def _libv2_root() -> Path:
    """Resolve the LibV2 root, honoring ``ED4ALL_LIBV2_ROOT``.

    Read at call time (not import time) so tests can redirect via the env
    var and so the module stays import-safe even if ``lib.paths`` is slow.
    """
    override = os.environ.get("ED4ALL_LIBV2_ROOT")
    if override:
        return Path(override)
    from lib.paths import LIBV2_PATH  # noqa: PLC0415

    return Path(LIBV2_PATH)


def _courses_dir() -> Path:
    return _libv2_root() / "courses"


def _course_dir(slug: str) -> Path:
    return _courses_dir() / slug


def _libv2_root_on_path() -> Path:
    """Ensure ``LibV2/`` is on ``sys.path`` so ``tools.libv2.*`` imports.

    ``LibV2/tools/`` ships no ``__init__.py``; the package is only addressable
    as the top-level ``tools.libv2`` namespace when its parent (``LibV2/``) is
    on ``sys.path``. This mirrors how the LibV2 CLI runs
    (``python -m tools.libv2.cli`` from the ``LibV2/`` directory). Returns the
    LibV2 root so callers can pass it as ``repo_root``.
    """
    import sys  # noqa: PLC0415

    root = _libv2_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _resolve_chunks_path(course_dir: Path) -> Optional[Path]:
    """Best-effort path to a course's ``chunks.jsonl`` (imscc, then legacy).

    Uses the canonical ``lib.libv2_storage.resolve_imscc_chunks_path`` shim
    when importable so the legacy ``corpus/`` fallback stays consistent with
    the retriever. Returns ``None`` when no readable chunkset exists.
    """
    try:
        from lib.libv2_storage import resolve_imscc_chunks_path  # noqa: PLC0415

        path = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
        if path.exists():
            return path
    except Exception:  # pragma: no cover - defensive
        pass
    # Direct fallbacks (covers the stale-empty-imscc_chunks/ corpus state).
    for candidate in (
        course_dir / "imscc_chunks" / "chunks.jsonl",
        course_dir / "dart_chunks" / "chunks.jsonl",
        course_dir / "corpus" / "chunks.jsonl",
    ):
        if candidate.exists():
            return candidate
    return None


def _count_chunks(course_dir: Path) -> int:
    """Chunk count for a course — manifest first (no token cost), then count.

    Reads ``manifest.json::content_profile.total_chunks`` when present (the
    cheap path the LibV2 token-cost policy prefers). Falls back to counting
    non-blank lines in ``chunks.jsonl`` only when the manifest lacks the
    field. Returns 0 when neither is resolvable.
    """
    manifest = course_dir / "manifest.json"
    if manifest.exists():
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            profile = doc.get("content_profile") or {}
            total = profile.get("total_chunks")
            if isinstance(total, int) and total >= 0:
                return total
        except (OSError, ValueError):
            pass
    chunks_path = _resolve_chunks_path(course_dir)
    if chunks_path is None:
        return 0
    try:
        count = 0
        with chunks_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    except OSError:
        return 0


# ------------------------------------------------------------------- courses


def list_courses() -> List[Dict[str, Any]]:
    """List LibV2 courses as ``[{slug, chunk_count}]``.

    A directory is treated as a course when it carries a ``manifest.json`` OR
    a resolvable ``chunks.jsonl`` — this skips scratch dirs that are neither.
    Sorted by slug for stable frontend rendering.
    """
    courses_dir = _courses_dir()
    if not courses_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for entry in sorted(courses_dir.iterdir()):
        if not entry.is_dir():
            continue
        has_manifest = (entry / "manifest.json").exists()
        has_chunks = _resolve_chunks_path(entry) is not None
        if not (has_manifest or has_chunks):
            continue
        out.append({"slug": entry.name, "chunk_count": _count_chunks(entry)})
    return out


# ----------------------------------------------------------------- retrieval


def _clamp_top_k(top_k: Optional[int]) -> int:
    """Clamp a requested ``top_k`` into ``[1, TOP_K_CAP]`` with a 5 default."""
    if top_k is None:
        return 5
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return 5
    if value < 1:
        return 1
    return min(value, TOP_K_CAP)


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Project a ``RetrievalResult`` (or dict) to the frontend result shape.

    The frontend contract wants at minimum ``score`` + ``text``; we also
    surface ``chunk_id`` + ``learning_outcome_refs`` + a few metadata axes so
    the Retrieval tab can render scores with provenance. ``rationale`` is
    carried through when present (it's only set with ``include_rationale``).
    """
    if isinstance(result, dict):
        src = result
    else:
        # RetrievalResult exposes to_dict(); prefer it for the canonical shape.
        to_dict = getattr(result, "to_dict", None)
        src = to_dict() if callable(to_dict) else result.__dict__
    out: Dict[str, Any] = {
        "score": float(src.get("score", 0.0) or 0.0),
        "text": src.get("text", ""),
        "chunk_id": src.get("chunk_id"),
        "course_slug": src.get("course_slug"),
        "chunk_type": src.get("chunk_type"),
        "concept_tags": src.get("concept_tags", []),
        "learning_outcome_refs": src.get("learning_outcome_refs", []),
        "bloom_level": src.get("bloom_level"),
        "tokens_estimate": src.get("tokens_estimate", 0),
    }
    if src.get("rationale") is not None:
        out["rationale"] = src.get("rationale")
    return out


# Filter keys we forward to ``retrieve_chunks`` — others are ignored so a
# noisy frontend payload can't crash the retriever with an unexpected kwarg.
_RETRIEVE_FILTER_KEYS = (
    "chunk_type",
    "difficulty",
    "concept_tags",
    "learning_outcome_refs",
    "bloom_level",
    "teaching_role",
    "content_type_label",
    "module_id",
    "week_num",
    "cognitive_domain",
    "hierarchy_level",
)


def retrieve(
    slug: str,
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """BM25 retrieval against one LibV2 course via the REAL retriever.

    Calls ``tools.libv2.retriever.retrieve_chunks`` scoped to ``slug`` with a
    clamped ``limit``. ``filters`` forwards the recognized ``ChunkFilter``
    axes (chunk_type, concept_tags, bloom_level, …). ``include_rationale`` is
    on so each result carries its BM25 / boost breakdown for the UI.
    """
    if not query or not query.strip():
        return []
    root = _libv2_root_on_path()
    from tools.libv2.retriever import retrieve_chunks  # noqa: PLC0415

    limit = _clamp_top_k(top_k)
    kwargs: Dict[str, Any] = {}
    for key in _RETRIEVE_FILTER_KEYS:
        value = (filters or {}).get(key)
        if value is not None and value != [] and value != "":
            kwargs[key] = value

    results = retrieve_chunks(
        repo_root=root,
        query=query,
        course_slug=slug,
        limit=limit,
        include_rationale=True,
        **kwargs,
    )
    return [_result_to_dict(r) for r in results]


def multi_retrieve(
    slug: str,
    query: str,
    explain: bool = False,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Multi-query (decomposition + RRF fusion) retrieval via REAL backend.

    Wraps ``MultiQueryRetriever.retrieve_with_decomposition`` and returns
    ``{results, decomposition}``. ``decomposition`` carries the sub-query
    plan so the UI can show how the query was split. The retriever's
    decomposition is course-agnostic at the API surface but we still pass the
    course context implicitly via the shared LibV2 repo root.
    """
    if not query or not query.strip():
        return {"results": [], "decomposition": None}
    root = _libv2_root_on_path()
    from tools.libv2.multi_retriever import MultiQueryRetriever  # noqa: PLC0415

    limit = _clamp_top_k(top_k)
    retriever = MultiQueryRetriever(repo_root=root)
    fusion_result, decomposed = retriever.retrieve_with_decomposition(
        query=query,
        limit=limit,
    )

    results = [_result_to_dict(r) for r in getattr(fusion_result, "results", [])]

    decomposition: Optional[Dict[str, Any]] = None
    if explain or decomposed is not None:
        decomposition = _decomposition_to_dict(decomposed, fusion_result)
    return {"results": results, "decomposition": decomposition}


def _decomposition_to_dict(decomposed: Any, fusion_result: Any) -> Dict[str, Any]:
    """Serialize a ``DecomposedQuery`` + fusion metadata to a plain dict."""
    out: Dict[str, Any] = {}
    if decomposed is not None:
        primary = getattr(decomposed, "primary_intent", None)
        out["primary_intent"] = getattr(primary, "value", primary)
        out["bloom_level"] = getattr(decomposed, "bloom_level", None)
        out["detected_concepts"] = getattr(decomposed, "detected_concepts", None)
        sub_queries = getattr(decomposed, "sub_queries", []) or []
        out["sub_queries"] = [
            {
                "text": getattr(sq, "text", ""),
                "aspect": getattr(getattr(sq, "aspect", None), "value", None),
                "weight": getattr(sq, "weight", None),
            }
            for sq in sub_queries
        ]
    out["fusion_method"] = getattr(fusion_result, "fusion_method", None)
    out["intent_coverage"] = getattr(fusion_result, "intent_coverage", None)
    return out


# ------------------------------------------------------------------ adapters


def _resolve_base_model_repo(model_card: Dict[str, Any]) -> Optional[str]:
    """Pull the HF base-model repo out of a model card (str or dict form)."""
    base = model_card.get("base_model")
    if isinstance(base, str):
        return base
    if isinstance(base, dict):
        return (
            base.get("huggingface_repo")
            or base.get("hf_repo")
            or base.get("repo")
        )
    return None


def list_adapters(slug: str) -> List[Dict[str, Any]]:
    """Scan ``LibV2/courses/<slug>/models/*`` for trained adapters.

    Reads ``model_card.json`` (base model repo) and, when present,
    ``eval_report.json`` (surfaces a couple of headline eval scores). A
    directory without a ``model_card.json`` is skipped — it isn't a promoted
    adapter. Sorted by ``model_id`` for stable rendering.
    """
    models_dir = _course_dir(slug) / "models"
    if not models_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        card_path = entry / "model_card.json"
        if not card_path.exists():
            continue
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            card = {}
        record: Dict[str, Any] = {
            "model_id": card.get("model_id") or entry.name,
            "base_model": _resolve_base_model_repo(card),
        }
        eval_path = entry / "eval_report.json"
        if eval_path.exists():
            try:
                report = json.loads(eval_path.read_text(encoding="utf-8"))
                record["eval"] = {
                    "faithfulness": report.get("faithfulness"),
                    "baseline_delta": report.get("baseline_delta"),
                    "smoke_mode": report.get("smoke_mode", False),
                }
            except (OSError, ValueError):
                pass
        out.append(record)
    return out


def _find_adapter_dir(slug: str, model_id: str) -> Optional[Path]:
    """Locate an adapter dir by ``model_id`` (dir name OR card ``model_id``)."""
    models_dir = _course_dir(slug) / "models"
    if not models_dir.exists():
        return None
    direct = models_dir / model_id
    if direct.is_dir() and (direct / "model_card.json").exists():
        return direct
    # Fall back to matching the card's model_id field.
    for entry in models_dir.iterdir():
        if not entry.is_dir():
            continue
        card_path = entry / "model_card.json"
        if not card_path.exists():
            continue
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if card.get("model_id") == model_id:
            return entry
    return None


def adapter_infer(
    slug: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Run REAL adapter inference via ``AdapterCallable``.

    Loads the base model + saved LoRA adapter from
    ``LibV2/courses/<slug>/models/<model_id>/`` (base repo read from
    ``model_card.json``) and returns ``{generation, base_model}``.

    Heavy ML deps (torch / transformers / peft / bitsandbytes) are imported
    LAZILY inside this function. On ``ImportError`` we return a typed
    ``{"error": "training_deps_missing", "detail": ...}`` dict — the router
    maps this to HTTP 503. We NEVER fabricate a generation.

    Other failure modes (missing adapter, unresolvable base repo, model-load
    error) return ``{"error": ..., "detail": ...}`` with a distinct code so
    the router can map them to 404 / 422 / 500 rather than 503.
    """
    if not prompt or not prompt.strip():
        return {"error": "invalid_prompt", "detail": "prompt must be non-empty"}

    adapter_dir = _find_adapter_dir(slug, model_id)
    if adapter_dir is None:
        return {
            "error": "adapter_not_found",
            "detail": (
                f"No adapter with model_id={model_id!r} under "
                f"course {slug!r}."
            ),
        }

    card_path = adapter_dir / "model_card.json"
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "error": "model_card_unreadable",
            "detail": f"Failed to read model_card.json: {exc}",
        }

    base_repo = _resolve_base_model_repo(card)
    if not base_repo:
        return {
            "error": "base_model_unresolved",
            "detail": (
                "model_card.json carries no resolvable base model repo "
                "(expected base_model as a string or "
                "base_model.huggingface_repo)."
            ),
        }

    # Lazy import: AdapterCallable's __init__ imports torch/transformers/peft/
    # bitsandbytes. A missing-deps box should get a typed 503, not a crash.
    try:
        from Trainforge.eval.adapter_callable import AdapterCallable  # noqa: PLC0415
    except ImportError as exc:
        return {
            "error": "training_deps_missing",
            "detail": (
                "Adapter inference requires the [training] extra "
                f"(torch/transformers/peft/bitsandbytes): {exc}"
            ),
        }

    try:
        callable_ = AdapterCallable(
            adapter_dir=adapter_dir,
            base_model_repo=base_repo,
            max_new_tokens=int(max_new_tokens),
            temperature=0.0,
        )
    except ImportError as exc:
        # The heavy imports live inside __init__ too.
        return {
            "error": "training_deps_missing",
            "detail": (
                "Adapter inference requires the [training] extra "
                f"(torch/transformers/peft/bitsandbytes): {exc}"
            ),
        }
    except FileNotFoundError as exc:
        return {"error": "adapter_incomplete", "detail": str(exc)}
    except (KeyError, RuntimeError) as exc:
        # KeyError: base model spec unresolvable. RuntimeError: known
        # transformers/accelerate incompatibility translated by AdapterCallable.
        return {"error": "adapter_load_failed", "detail": str(exc)}

    try:
        generation = callable_(prompt)
    except Exception as exc:  # pragma: no cover - generation runtime errors
        return {"error": "inference_failed", "detail": str(exc)}

    return {"generation": generation, "base_model": base_repo}


# --------------------------------------------------------------- LLM rerank


_RERANK_SYSTEM = (
    "You are a retrieval reranker. Given a query and a numbered list of "
    "candidate passages, return ONLY a JSON array. Each element is an object "
    '{"index": <int>, "rationale": "<one sentence why this passage is '
    'relevant or not>"} ordered from MOST to LEAST relevant to the query. '
    "Use the candidate's original index. Do not invent passages, do not omit "
    "any candidate, and do not output anything except the JSON array."
)


def _build_rerank_prompt(query: str, results: List[Dict[str, Any]]) -> str:
    """Build the user prompt enumerating BM25 candidates for the reranker."""
    lines = [f"Query: {query}", "", "Candidates:"]
    for idx, result in enumerate(results):
        text = (result.get("text") or "").strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + "…"
        lines.append(f"[{idx}] {text}")
    lines.append("")
    lines.append(
        "Return the JSON array ordering these candidates by relevance, "
        "with a one-sentence rationale per candidate."
    )
    return "\n".join(lines)


def _parse_rerank_response(raw: str, n: int) -> Optional[List[Dict[str, Any]]]:
    """Parse the reranker's JSON array into ``[{index, rationale}]``.

    Tolerates surrounding prose by extracting the first ``[...]`` span.
    Returns ``None`` when no valid ordering is recoverable (the caller then
    returns a typed error rather than a fabricated order).
    """
    if not raw:
        return None
    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    order: List[Dict[str, Any]] = []
    seen: set = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= n or idx in seen:
            continue
        seen.add(idx)
        order.append({"index": idx, "rationale": str(item.get("rationale") or "")})
    return order or None


def llm_rerank(
    slug: str,
    query: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Custom LLM-driven rerank of BM25 candidates via the REAL backend.

    Applies the active GUI settings (provider/model/keys) to the environment,
    builds a rerank prompt over the supplied candidates, and calls
    ``MCP.orchestrator.llm_backend.build_backend(...).complete_sync(...)``.
    Returns ``{"results": [...reordered with per-item rationale...]}``.

    On backend / auth / parse failure returns a typed
    ``{"error": ..., "detail": ...}`` dict — NEVER a fake ranking. The
    ``slug`` is carried for symmetry / future per-course routing; the rerank
    itself operates purely on the passed candidates.
    """
    if not results:
        return {"results": []}
    if not query or not query.strip():
        return {"error": "invalid_query", "detail": "query must be non-empty"}

    # Apply the active settings so build_backend picks up provider/model/keys.
    try:
        from gui import settings_store  # noqa: PLC0415

        settings_store.apply_env(settings_store.load_settings())
    except Exception as exc:  # pragma: no cover - settings are best-effort
        return {
            "error": "settings_apply_failed",
            "detail": f"Could not apply GUI settings before rerank: {exc}",
        }

    try:
        from MCP.orchestrator.llm_backend import build_backend  # noqa: PLC0415
    except ImportError as exc:
        return {
            "error": "backend_unavailable",
            "detail": f"LLM backend module not importable: {exc}",
        }

    try:
        backend = build_backend()
    except Exception as exc:
        return {"error": "backend_build_failed", "detail": str(exc)}

    prompt = _build_rerank_prompt(query, results)
    try:
        raw = backend.complete_sync(
            system=_RERANK_SYSTEM,
            user=prompt,
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as exc:
        # Covers auth failures, local-mode throwing backend, network errors.
        return {"error": "rerank_call_failed", "detail": str(exc)}

    order = _parse_rerank_response(raw if isinstance(raw, str) else "", len(results))
    if order is None:
        return {
            "error": "rerank_parse_failed",
            "detail": "LLM did not return a parseable rerank ordering.",
        }

    reranked: List[Dict[str, Any]] = []
    for rank, item in enumerate(order):
        base = dict(results[item["index"]])
        base["rationale"] = item["rationale"]
        base["rerank_position"] = rank
        reranked.append(base)
    # Append any candidates the model dropped, preserving original order, so
    # the caller always gets every passed candidate back.
    ranked_indices = {item["index"] for item in order}
    for idx, result in enumerate(results):
        if idx not in ranked_indices:
            tail = dict(result)
            tail.setdefault("rationale", "")
            reranked.append(tail)
    return {"results": reranked}
