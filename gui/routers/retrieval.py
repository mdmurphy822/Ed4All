"""Retrieval REST router — ``/api/retrieval/*`` (BM25/multi/LLM-rerank + adapter infer).

Mounted at prefix ``/api/retrieval`` by ``gui.app``, so the paths here are
relative. The router is thin: all real backend wiring lives in
``gui.services.retrieval_service`` (which imports without FastAPI / torch /
LibV2 — heavy imports are lazy inside the service functions).

Endpoint contract (frontend-facing):

* ``GET  "/courses"``        → ``[{slug, chunk_count}]`` from LibV2 manifests.
* ``POST "/query"``          → BM25 / multi / LLM-rerank retrieval, routed by ``mode``.
* ``GET  "/{slug}/adapters"`` → ``[{model_id, base_model?}]`` for a course.
* ``POST "/{slug}/infer"``   → REAL adapter inference, or 503 when ML deps absent.

Errors surface as a typed ``{error, detail}`` body with the right status code
(422 bad input, 503 training deps missing, 404 not found, 500 unexpected) —
never a fabricated success.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gui.services import retrieval_service

router = APIRouter()


# ----------------------------------------------------------------- request models


class QueryRequest(BaseModel):
    """Body for ``POST "/query"``.

    ``mode`` routes to the backend: ``bm25`` → single-course BM25,
    ``multi`` → query-decomposition + RRF fusion, ``llm_rerank`` → BM25
    candidates then a live LLM rerank.
    """

    slug: str
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = Field(default_factory=dict)
    mode: str = "bm25"

    model_config = {"extra": "allow"}


class InferRequest(BaseModel):
    """Body for ``POST "/{slug}/infer"`` — adapter id + prompt + generation cap."""

    model_id: str
    prompt: str
    max_new_tokens: int = 256


# ------------------------------------------------------------------- error helper


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


# Service error codes that map to a non-500 HTTP status. Anything else from a
# service ``{"error": ...}`` payload is treated as an unexpected 500.
_ERROR_STATUS = {
    "training_deps_missing": 503,
    "backend_unavailable": 503,
    "adapter_load_failed": 503,  # transient: weights present but load failed (OOM/deps)
    "adapter_not_found": 404,
    "invalid_prompt": 422,
    "invalid_query": 422,
    "base_model_unresolved": 422,
    "model_card_unreadable": 422,
    "adapter_incomplete": 422,
    "inference_failed": 502,  # upstream generation failed after a successful load
}


# ----------------------------------------------------------------------- endpoints


@router.get("/courses")
async def get_courses() -> Any:
    """List LibV2 courses with chunk counts (``[{slug, chunk_count}]``)."""
    try:
        return retrieval_service.list_courses()
    except Exception as exc:  # noqa: BLE001 — unexpected filesystem/parse failure
        return _error(500, "list_courses_failed", str(exc))


@router.post("/query")
async def query(req: QueryRequest) -> Any:
    """Run a retrieval query, routed by ``mode`` (bm25 / multi / llm_rerank).

    * ``bm25`` → ``{results}``.
    * ``multi`` → ``{results, decomposition}``.
    * ``llm_rerank`` → BM25 retrieve, then a live LLM rerank → ``{results}``
      with a per-item ``rationale``. A rerank backend / auth / parse failure
      surfaces as a typed error (not a fabricated order).
    """
    mode = (req.mode or "bm25").lower()
    if not req.query or not req.query.strip():
        return _error(422, "invalid_query", "query must be non-empty")

    try:
        if mode == "bm25":
            results = retrieval_service.retrieve(
                req.slug, req.query, req.top_k, req.filters,
            )
            return {"results": results}

        if mode == "multi":
            return retrieval_service.multi_retrieve(
                req.slug, req.query, explain=True, top_k=req.top_k, filters=req.filters,
            )

        if mode == "llm_rerank":
            candidates = retrieval_service.retrieve(
                req.slug, req.query, req.top_k, req.filters,
            )
            reranked = retrieval_service.llm_rerank(req.slug, req.query, candidates)
            if isinstance(reranked, dict) and reranked.get("error"):
                status = _ERROR_STATUS.get(reranked["error"], 502)
                return _error(status, reranked["error"], reranked.get("detail", ""))
            return {"results": reranked.get("results", [])}

        return _error(
            422,
            "invalid_mode",
            f"mode must be one of bm25|multi|llm_rerank, got {req.mode!r}",
        )
    except ImportError as exc:
        return _error(503, "retriever_unavailable", str(exc))
    except Exception as exc:  # noqa: BLE001 — unexpected retrieval failure
        return _error(500, "query_failed", str(exc))


@router.get("/{slug}/adapters")
async def get_adapters(slug: str) -> List[Dict[str, Any]]:
    """List trained adapters for a course (``[{model_id, base_model?, eval?}]``)."""
    try:
        return retrieval_service.list_adapters(slug)
    except Exception as exc:  # noqa: BLE001 — unexpected scan/parse failure
        return _error(500, "list_adapters_failed", str(exc))  # type: ignore[return-value]


@router.post("/{slug}/infer")
async def infer(slug: str, req: InferRequest) -> Any:
    """Run REAL adapter inference; 503 when the ML deps are missing.

    Returns ``{generation, base_model}`` on success. Heavy-deps absence
    yields ``{error: "training_deps_missing", detail}`` mapped to HTTP 503 —
    never a fabricated generation. Missing adapter → 404; bad input → 422.
    """
    try:
        result = retrieval_service.adapter_infer(
            slug, req.model_id, req.prompt, req.max_new_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — unexpected load/inference failure
        return _error(500, "infer_failed", str(exc))

    if isinstance(result, dict) and result.get("error"):
        status = _ERROR_STATUS.get(result["error"], 500)
        return _error(status, result["error"], result.get("detail", ""))
    return result


__all__ = ["router"]
