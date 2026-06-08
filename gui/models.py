"""Pydantic request/response models shared by the GUI routers (spec §6).

Pydantic-only — NO FastAPI import — so MCP tools / non-web code can import the
shapes without pulling web deps. Pydantic v2 is a default project dep.

All error responses use ``ErrorResponse`` (``{error, detail}``) per the §6
contract; never a fabricated success.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ----------------------------------------------------------------- common


class ErrorResponse(BaseModel):
    """Canonical error envelope: ``{error, detail}`` (spec §6)."""

    error: str
    detail: str = ""


class OkResponse(BaseModel):
    """Trivial health/ack response."""

    status: str = "ok"


# ----------------------------------------------------------------- settings


class SettingsResponse(BaseModel):
    """GET /api/settings — masked settings doc + catalog + providers + models."""

    settings: Dict[str, Any]
    catalog: List[Dict[str, Any]] = Field(default_factory=list)
    providers: List[Dict[str, Any]] = Field(default_factory=list)
    base_models: List[str] = Field(default_factory=list)


class SettingsDoc(BaseModel):
    """Full settings doc for PUT /api/settings.

    Permissive by design — the canonical validation lives in
    ``settings_store._validate`` (catalog-aware), not in the wire model.
    """

    version: int = 1
    updated_at: Optional[str] = None
    env: Dict[str, Any] = Field(default_factory=dict)
    model_routing: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    flags: Dict[str, Any] = Field(default_factory=dict)


class SettingsPatch(BaseModel):
    """PATCH /api/settings — partial deep-merge patch (single-field saves)."""

    env: Optional[Dict[str, Any]] = None
    model_routing: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None
    flags: Optional[Dict[str, Any]] = None


class ApplyEnvResponse(BaseModel):
    """POST /api/settings/apply — applied env keys (secret values masked)."""

    applied: Dict[str, str] = Field(default_factory=dict)


class TestProviderRequest(BaseModel):
    """POST /api/settings/test-provider — reachability check for one provider."""

    provider: str


class TestProviderResponse(BaseModel):
    """Result of a live provider reachability check (real, not stubbed)."""

    provider: str
    reachable: bool
    detail: str = ""


# ----------------------------------------------------------------- uploads


class UploadedFile(BaseModel):
    """One saved file inside an upload bundle."""

    name: str
    path: str
    size: int
    kind: str


class UploadResponse(BaseModel):
    """POST /api/uploads — saved corpus bundle."""

    upload_id: str
    files: List[UploadedFile] = Field(default_factory=list)


class UploadSummary(BaseModel):
    """GET /api/uploads — listing entry."""

    upload_id: str
    files: List[UploadedFile] = Field(default_factory=list)
    created_at: Optional[str] = None


# ------------------------------------------------------------------- runs


class PhaseInfo(BaseModel):
    """One phase within a workflow descriptor."""

    name: str
    agents: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    validation_gates_count: int = 0


class WorkflowInfo(BaseModel):
    """GET /api/workflows — one workflow descriptor."""

    name: str
    phases: List[PhaseInfo] = Field(default_factory=list)


class LaunchRunRequest(BaseModel):
    """POST /api/runs — launch a full workflow."""

    workflow: str
    course_name: str
    corpus: Optional[str] = None  # upload_id or absolute path
    weeks: Optional[int] = None
    mode: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class LaunchPhaseRequest(BaseModel):
    """POST /api/runs/phase — launch a single phase against an existing export."""

    workflow: str
    phase: str
    course_name: str
    project_id: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    """Launch acknowledgement."""

    run_id: str
    workflow_id: Optional[str] = None
    status: str


class RunStatus(BaseModel):
    """GET /api/runs/{run_id} — run registry + live status."""

    run_id: str
    workflow: Optional[str] = None
    workflow_id: Optional[str] = None
    status: str = "unknown"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    phase: Optional[str] = None
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    gate_results: List[Dict[str, Any]] = Field(default_factory=list)
    detail: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- courses


class CourseSummary(BaseModel):
    """GET /api/courses — one course summary."""

    course_name: str
    project_id: Optional[str] = None
    slug: Optional[str] = None
    duration_weeks: Optional[int] = None
    terminal_count: int = 0
    chapter_count: int = 0
    topics: List[str] = Field(default_factory=list)
    subdomains: List[str] = Field(default_factory=list)


class ObjectivesDoc(BaseModel):
    """GET/PUT /api/courses/{id}/objectives — synthesized objectives doc.

    Permissive container; the LO-id pattern validation lives in the service
    layer (``lib.ontology.learning_objectives.validate_lo_id``).
    """

    terminal_objectives: List[Dict[str, Any]] = Field(default_factory=list)
    chapter_objectives: List[Dict[str, Any]] = Field(default_factory=list)
    mint_method: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ClassificationDoc(BaseModel):
    """GET /api/courses/{id}/classification — LibV2 manifest classification."""

    division: Optional[str] = None
    primary_domain: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    subtopics: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ClassificationPatch(BaseModel):
    """PATCH /api/courses/{id}/classification — partial classification edit."""

    division: Optional[str] = None
    primary_domain: Optional[str] = None
    topics: Optional[List[str]] = None
    subtopics: Optional[List[str]] = None
    tags: Optional[List[str]] = None


# --------------------------------------------------------------- retrieval


class RetrievalCourse(BaseModel):
    """GET /api/retrieval/courses — slug + chunk count."""

    slug: str
    chunk_count: int = 0


class RetrievalQuery(BaseModel):
    """POST /api/retrieval/query — BM25 / multi / llm_rerank query."""

    slug: str
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = Field(default_factory=dict)
    mode: str = "bm25"  # bm25 | multi | llm_rerank


class RetrievalResultItem(BaseModel):
    """One retrieval hit mapped from a ``RetrievalResult``."""

    chunk_id: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """POST /api/retrieval/query — ranked results."""

    slug: str
    mode: str
    results: List[RetrievalResultItem] = Field(default_factory=list)
    explain: Dict[str, Any] = Field(default_factory=dict)


class AdapterSummary(BaseModel):
    """GET /api/retrieval/{slug}/adapters — one trained adapter."""

    model_id: str
    base_model: Optional[str] = None
    eval_scores: Dict[str, Any] = Field(default_factory=dict)
    is_current: bool = False


class InferRequest(BaseModel):
    """POST /api/retrieval/{slug}/infer — adapter inference."""

    model_id: str
    prompt: str
    max_new_tokens: int = 256


class InferResponse(BaseModel):
    """Adapter inference result."""

    model_id: str
    prompt: str
    generation: str


# ------------------------------------------------------------- events / bridge


class EventRecord(BaseModel):
    """One Claude<->GUI bridge event."""

    seq: int
    source: str
    kind: str
    ts: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class PostEventRequest(BaseModel):
    """POST a ``gui``-sourced event into the Activity bridge."""

    kind: str
    payload: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ErrorResponse",
    "OkResponse",
    "SettingsResponse",
    "SettingsDoc",
    "SettingsPatch",
    "ApplyEnvResponse",
    "TestProviderRequest",
    "TestProviderResponse",
    "UploadedFile",
    "UploadResponse",
    "UploadSummary",
    "PhaseInfo",
    "WorkflowInfo",
    "LaunchRunRequest",
    "LaunchPhaseRequest",
    "RunResponse",
    "RunStatus",
    "CourseSummary",
    "ObjectivesDoc",
    "ClassificationDoc",
    "ClassificationPatch",
    "RetrievalCourse",
    "RetrievalQuery",
    "RetrievalResultItem",
    "RetrievalResponse",
    "AdapterSummary",
    "InferRequest",
    "InferResponse",
    "EventRecord",
    "PostEventRequest",
]
