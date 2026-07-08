"""Learner answer REST router — ``/api/learn/*`` (ask, courses, source).

Mounted at prefix ``/api/learn`` by ``gui.app``, so the paths here are relative.
Thin router: the grounded-answer wiring lives in ``gui.services.answer_service``
(lazy heavy imports), the server-side HTML rendering in
``gui.services.answer_render``, and the archived-source-page resolution +
sanitisation in ``gui.services.source_page`` (a sibling executor's module; this
router codes against its FROZEN signature and imports it lazily so the router
mounts cleanly even while that module is mid-build).

Endpoint contract (learner-facing):

* ``GET  "/courses"``        → ``[{slug, chunk_count}]`` (delegates the operator
  ``retrieval_service.list_courses``).
* ``POST "/ask"``            → ``{answer: GroundedAnswer.to_dict(), html: str}``
  with HTTP **200 for ALL six pipeline statuses** — a refusal or a citation-gate
  block is a successful pipeline *outcome*, i.e. data, not an HTTP error. Typed
  backend errors (engine down / index missing / compose failure) map to the
  503/502/500 contract below and ALSO carry a rendered learner-safe ``html``
  fragment so the JS swap path is uniform.
* ``GET  "/source/{slug}"``  → the archived course page wrapped in a viewer shell
  with a restrictive CSP, for citation-back navigation.

Errors surface as a typed ``{error, detail}`` body (plus ``html`` for the ask
path) with the right status — never a fabricated success / answer.
"""

from __future__ import annotations

import json
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from gui import shared_state
from gui.services import answer_render, answer_service

router = APIRouter()


# ----------------------------------------------------------------- request models


class AskRequest(BaseModel):
    """Body for ``POST "/ask"``.

    ``engine ∈ {auto, lexical, semantic, hybrid-rrf}`` (default ``"auto"``,
    resolved honestly in the service: ``auto`` → ``hybrid-rrf`` when a vector
    index exists, else ``lexical``). Mirrors the operator ``QueryRequest``
    permissive ``extra="allow"`` posture.
    """

    slug: str
    query: str
    engine: str = "auto"
    # L3: request-level "search all courses" override. ``None`` => resolve from
    # ``ED4ALL_ANSWER_LIBRARY_WIDE`` (env default); an explicit bool WINS over it.
    library_wide: Optional[bool] = None

    model_config = {"extra": "allow"}


# ------------------------------------------------------------------- error helper


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(
        status_code=status_code, content={"error": error, "detail": detail}
    )


def _error_with_fragment(
    status_code: int, error: str, copy_key: str, detail: str
) -> JSONResponse:
    """Typed ``{error, detail, html}`` error for the ask path.

    Carries a rendered learner-safe fragment (``copy_key`` → ``ERROR_COPY``) so
    the JS swap path is identical on success and failure: it always inserts
    ``body.html``; only the HTTP status drives the live-region message. The
    operator-actionable ``detail`` is JSON-only and never rendered.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "html": answer_render.render_error_fragment(copy_key),
        },
    )


# Exception ``__class__.__name__`` → (http_status, error_code, copy_key). Keyed
# by name so the router needs NO heavy imports (LibV2 / Trainforge) to match a
# typed error class — the WS3 § 3.4 typed-error contract.
_TYPED_ERROR_MAP: Dict[str, Any] = {
    "AnswerBackendUnavailable": (503, "answer_backend_unavailable", "error_backend_down"),
    "AnswerProviderNotLocal": (503, "answer_provider_not_local", "error_misconfigured"),
    "AnswerComposeError": (502, "answer_compose_failed", "error_compose"),
    "InvalidCitationError": (502, "answer_compose_failed", "error_compose"),
    "SemanticIndexMissing": (503, "semantic_index_unavailable", "error_index"),
    "SemanticIndexStale": (503, "semantic_index_unavailable", "error_index"),
    "EmbeddingBackendUnavailable": (503, "embedding_backend_unavailable", "error_index"),
}


def _course_exists(slug: str) -> bool:
    """Whether ``slug`` is a known LibV2 course (404 gate; never 404 a 0-chunk course).

    Keyed on ``retrieval_service.list_courses()`` (the operator-service precedent)
    — a known slug with zero chunks still exists.
    """
    from gui.services import retrieval_service  # noqa: PLC0415

    try:
        return any(c.get("slug") == slug for c in retrieval_service.list_courses())
    except Exception:  # noqa: BLE001 — unknown listing failure → treat as unknown
        return False


# ----------------------------------------------------------------------- endpoints


@router.get("/courses")
async def get_courses() -> Any:
    """List LibV2 courses with chunk counts (``[{slug, chunk_count}]``)."""
    from gui.services import retrieval_service  # noqa: PLC0415

    try:
        return retrieval_service.list_courses()
    except Exception as exc:  # noqa: BLE001 — unexpected filesystem/parse failure
        return _error(500, "list_courses_failed", str(exc))


@router.post("/ask")
async def ask(req: AskRequest) -> Any:
    """Answer a learner question; 200 + rendered fragment for all six statuses.

    Refusals and citation-gate blocks are 200s (the pipeline succeeded; the
    *answer* outcome is data). Typed backend errors map per ``_TYPED_ERROR_MAP``
    to 503/502, anything else to 500 — each error response still carries a
    rendered learner-safe ``html`` fragment.
    """
    if not req.query or not req.query.strip():
        return _error_with_fragment(
            422, "invalid_query", "error_generic", "query must be non-empty"
        )
    if not _course_exists(req.slug):
        return _error_with_fragment(
            404, "course_not_found", "error_generic", f"unknown course: {req.slug!r}"
        )

    # ``library_wide`` threaded ONLY when the request set it explicitly, so the
    # default path stays byte-identical (a 3-arg ``answer_service.ask`` seam is
    # unchanged; the request value wins over the env only when present).
    ask_kwargs: Dict[str, Any] = {}
    if req.library_wide is not None:
        ask_kwargs["library_wide"] = req.library_wide
    try:
        answer = answer_service.ask(req.slug, req.query, req.engine, **ask_kwargs)
    except Exception as exc:  # noqa: BLE001 — map typed pipeline errors by class name
        name = type(exc).__name__
        if name in _TYPED_ERROR_MAP:
            status_code, error_code, copy_key = _TYPED_ERROR_MAP[name]
            return _error_with_fragment(status_code, error_code, copy_key, str(exc))
        # A pre-E3 ``_retrieve`` guard raises a bare RuntimeError naming the
        # missing engine dependency — surface as an index-unavailable 503.
        if name == "RuntimeError":
            return _error_with_fragment(
                503, "engine_unavailable", "error_index", str(exc)
            )
        return _error_with_fragment(500, "ask_failed", "error_generic", str(exc))

    include_links = answer_service.source_materials_enabled(req.slug)
    return {
        "answer": answer,
        "html": answer_render.render_answer_fragment(
            answer, include_original_source_links=include_links
        ),
    }


# --------------------------------------------------------------- async ask jobs


@router.get("/ask-ready/{slug}")
async def ask_ready(slug: str) -> Any:
    """Cheap server check: is ``slug`` a known course (the Ask drawer gate)?

    Returns ``{slug, exists, has_vector_index}`` — the drawer gates its ask box
    on ``exists`` (the course is reachable at all) and surfaces ``has_vector_index``
    so the "no index → explain how to fix it" copy keys off a server fact, not
    only the library card. ``has_vector_index`` is a single ``stat`` (no model
    load); ``False`` does NOT block asking (the engine honestly downgrades to
    lexical), it only drives the explanatory copy.
    """
    if not _course_exists(slug):
        return _error(404, "course_not_found", f"unknown course: {slug!r}")
    from gui.services.answer_service import _has_vector_index, _libv2_root  # noqa: PLC0415

    try:
        has_index = _has_vector_index(_libv2_root(), slug)
    except Exception:  # noqa: BLE001 — a stat failure → treat as no index (honest downgrade)
        has_index = False
    return {"slug": slug, "exists": True, "has_vector_index": has_index}


@router.get("/ask-capabilities")
async def ask_capabilities() -> Any:
    """Cheap capability probe for the Ask drawer (L3 library-wide gating).

    Returns ``{indexed_course_count, library_wide_eligible, library_wide_default}``.
    ``library_wide_eligible`` is true only when MORE THAN ONE course carries a
    vector index (a single-course library has nothing to union), so the drawer's
    "search all courses" control is shown only when it would do something.
    ``library_wide_default`` surfaces the resolved ``ED4ALL_ANSWER_LIBRARY_WIDE``
    env default so the checkbox can seed its initial state. All fs stats, no model
    load; a scan failure degrades to "not eligible".
    """
    from gui.services.answer_service import _libv2_root  # noqa: PLC0415
    from lib.retrieval.library_wide import resolve_library_wide  # noqa: PLC0415

    indexed = 0
    try:
        courses_root = _libv2_root() / "courses"
        if courses_root.is_dir():
            for child in sorted(courses_root.iterdir()):
                if child.is_dir() and (
                    child / "vector_index" / "manifest.json"
                ).is_file():
                    indexed += 1
    except Exception:  # noqa: BLE001 — a scan failure → not eligible (honest)
        indexed = 0

    default_on = False
    try:
        default_on = bool(resolve_library_wide(None))
    except Exception:  # noqa: BLE001 — env parse failure → off
        default_on = False

    return {
        "indexed_course_count": indexed,
        "library_wide_eligible": indexed > 1,
        "library_wide_default": default_on,
    }


@router.post("/ask-jobs")
async def submit_ask_job(req: AskRequest) -> Any:
    """Enqueue a durable async ask job; return ``{ask_id, status, queue_position}``.

    The async sibling of ``POST /ask`` for the in-context drawer: the answer is
    computed off-request by a single-lane background worker and persisted to disk
    (survives a tab refresh / uvicorn restart). Same 422/404 input gates as the
    sync path; the poll endpoint surfaces the result.
    """
    if not req.query or not req.query.strip():
        return _error(422, "invalid_query", "query must be non-empty")
    if not _course_exists(req.slug):
        return _error(404, "course_not_found", f"unknown course: {req.slug!r}")

    from gui.services import ask_jobs  # noqa: PLC0415

    submit_kwargs: Dict[str, Any] = {}
    if req.library_wide is not None:
        submit_kwargs["library_wide"] = req.library_wide
    record = ask_jobs.submit(req.slug, req.query, req.engine, **submit_kwargs)
    return {
        "ask_id": record["ask_id"],
        "status": record["status"],
        "queue_position": record.get("queue_position", -1),
    }


@router.get("/ask-jobs/{ask_id}")
async def poll_ask_job(ask_id: str) -> Any:
    """Poll one ask job → ``{status, ...}``.

    * ``pending`` / ``running`` → ``{ask_id, status, queue_position}``.
    * ``done`` → ``{ask_id, status, answer, html}`` (the answer rendered through
      the SAME ``answer_render`` path as the sync endpoint — one rendering path,
      no JS-vs-server drift).
    * ``error`` → ``{ask_id, status, error, detail, html}`` mapping the captured
      typed-error class name to a learner-safe fragment via ``_TYPED_ERROR_MAP``.

    Always HTTP 200 for a known job (the job *state* is data); an unknown id is
    404. Mirrors the sync ``/ask`` contract: refusals/blocks are ``done`` jobs
    carrying a refusal/block answer payload, never an HTTP error.
    """
    from gui.services import ask_jobs  # noqa: PLC0415

    record = ask_jobs.status(ask_id)
    if record is None:
        return _error(404, "ask_job_not_found", f"unknown ask job: {ask_id!r}")

    status_value = record.get("status")
    if status_value == ask_jobs.STATUS_DONE:
        answer = record.get("answer") or {}
        include_links = answer_service.source_materials_enabled(
            str(answer.get("course_slug") or record.get("slug") or "")
        )
        return {
            "ask_id": ask_id,
            "status": status_value,
            "answer": answer,
            "html": answer_render.render_answer_fragment(
                answer, include_original_source_links=include_links
            ),
        }
    if status_value == ask_jobs.STATUS_ERROR:
        name = str(record.get("error") or "")
        detail = str(record.get("detail") or "")
        if name in _TYPED_ERROR_MAP:
            _, error_code, copy_key = _TYPED_ERROR_MAP[name]
        elif name == "RuntimeError":
            error_code, copy_key = "engine_unavailable", "error_index"
        else:
            error_code, copy_key = "ask_failed", "error_generic"
        return {
            "ask_id": ask_id,
            "status": status_value,
            "error": error_code,
            "detail": detail,
            "html": answer_render.render_error_fragment(copy_key),
        }
    # pending / running
    resp: Dict[str, Any] = {
        "ask_id": ask_id,
        "status": status_value,
        "queue_position": record.get("queue_position", -1),
    }
    # L4 passages-first: once the pipeline has retrieved passages (persisted onto
    # the running record after the refusal gate), surface them so a poll can
    # disclose "here is what I found" while the compose call runs.
    passages = record.get("passages")
    if passages:
        resp["passages"] = passages
        resp["passages_refused"] = bool(record.get("passages_refused"))
    return resp


# ------------------------------------------------------- answer feedback (I6)
#
# Learner-open thumbs up/down (+ optional comment) on a completed grounded
# answer. The signal fans out to (1) the Claude<->GUI event log
# (``append_event('gui', 'answer_feedback', ...)``, so a Claude session sees it
# via ``gui_read_events``) and (2) a per-course ``feedback.jsonl`` (best-effort;
# an absent course dir is tolerated). No new decision_type — the training
# round-trip is deferred. The endpoint is no-auth, so the comment is size-capped
# and a coarse in-process rate limit damps floods (best-effort, not a security
# boundary).

_FEEDBACK_COMMENT_MAX = 2000  # chars; longer comments are truncated on store
_FEEDBACK_RATE_MAX = 30  # accepted submissions ...
_FEEDBACK_RATE_WINDOW_S = 60.0  # ... per this sliding window (per process)
_feedback_hits: "deque[float]" = deque()
_feedback_lock = threading.Lock()


def _feedback_rate_ok() -> bool:
    """Coarse per-process sliding-window admission (best-effort flood damping)."""
    now = time.monotonic()
    with _feedback_lock:
        while _feedback_hits and now - _feedback_hits[0] > _FEEDBACK_RATE_WINDOW_S:
            _feedback_hits.popleft()
        if len(_feedback_hits) >= _FEEDBACK_RATE_MAX:
            return False
        _feedback_hits.append(now)
        return True


def _append_course_feedback(slug: str, payload: Dict[str, Any]) -> None:
    """Append the feedback record to ``courses/<slug>/feedback.jsonl`` (best-effort).

    An absent course dir is tolerated (no dir is created); a path-unsafe slug is
    rejected before any filesystem touch. A write failure never surfaces to the
    caller — the event-log fan-out is the authoritative signal.
    """
    if "/" in slug or "\\" in slug or ".." in slug:
        return
    try:
        from gui.services.answer_service import _libv2_root  # noqa: PLC0415

        course_dir = _libv2_root() / "courses" / slug
        if not course_dir.is_dir():
            return  # absent course dir tolerated
        target = course_dir / "feedback.jsonl"
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:  # noqa: BLE001 — per-course persistence is advisory
        pass


class FeedbackRequest(BaseModel):
    """Body for ``POST /api/learn/feedback``.

    ``verdict ∈ {up, down}`` (required). ``ask_id`` (optional) ties the feedback
    to the durable ask job it rates; ``comment`` (optional) is free text,
    size-capped on store. Permissive ``extra="allow"`` mirrors ``AskRequest``.
    """

    slug: str
    verdict: str
    ask_id: Optional[str] = None
    comment: Optional[str] = None

    model_config = {"extra": "allow"}


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest) -> Any:
    """Record thumbs up/down (+ optional comment) on a completed answer (I6).

    Fans out to the event log (``answer_feedback``) AND a per-course
    ``feedback.jsonl`` (best-effort). Rejects an unknown verdict (422) and a
    flood (429); the comment is trimmed to ``_FEEDBACK_COMMENT_MAX`` chars.
    """
    verdict = (req.verdict or "").strip().lower()
    if verdict not in ("up", "down"):
        return _error(422, "invalid_verdict", "verdict must be 'up' or 'down'")
    if not req.slug or not req.slug.strip():
        return _error(422, "invalid_slug", "slug must be non-empty")
    if not _feedback_rate_ok():
        return _error(
            429, "rate_limited", "too many feedback submissions; retry shortly"
        )

    comment = req.comment if isinstance(req.comment, str) else ""
    comment = comment.strip()[:_FEEDBACK_COMMENT_MAX]

    payload: Dict[str, Any] = {
        "slug": req.slug,
        "verdict": verdict,
        "ask_id": (req.ask_id or None),
        "comment": comment,
        "ts": shared_state.now_iso(),
    }

    try:
        shared_state.append_event("gui", "answer_feedback", payload)
    except Exception:  # noqa: BLE001 — event-log append is best-effort
        pass
    _append_course_feedback(req.slug, payload)

    return {"ok": True, "verdict": verdict}


@router.get("/source/{slug}")
async def get_source(slug: str, item_path: str = "", fragment: str = "") -> Any:
    """Serve an archived course page wrapped in the viewer shell (citation-back).

    Thin delegate to the frozen ``source_page.render_source_page`` contract; that
    service owns path-safety (sanitise → resolve → contain), cartridge HTML
    sanitisation, heading-id injection, and the CSP / nosniff headers. The import
    is lazy so the router mounts even while that module is mid-build.
    """
    try:
        from gui.services import source_page  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "source_viewer_unavailable", str(exc))

    if not item_path:
        return _error(422, "invalid_item_path", "item_path is required")

    from gui.services.answer_service import _libv2_root  # noqa: PLC0415

    try:
        result = source_page.render_source_page(
            slug,
            item_path,
            _libv2_root(),
            fragment=fragment or None,
        )
    except source_page.SourcePageError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected resolution/render failure
        return _error(500, "source_render_failed", str(exc))

    return Response(
        content=result.html,
        media_type=result.media_type,
        headers=dict(result.headers),
    )


# --------------------------------------------------------------- quiz (W10 §5.1)


def _resolve_learner_id(raw: Optional[str]) -> str:
    """Normalize the ``X-Learner-Id`` owner key, or ``""`` when absent/unsafe.

    The id is an OPAQUE per-browser owner key (a localStorage uuid) — NOT an auth
    principal (W10 identity model; learner-open per §1.5). It is never trusted for
    anything beyond "owns their own records". A separator-bearing / over-long id is
    treated as absent (the persistence layer would reject it anyway) so a hostile
    header can't influence a path. Returns the stripped id or ``""``.
    """
    if not raw or not isinstance(raw, str):
        return ""
    candidate = raw.strip()
    if not candidate or len(candidate) > 128:
        return ""
    if "/" in candidate or "\\" in candidate or ".." in candidate:
        return ""
    return candidate


class GradeRequest(BaseModel):
    """Body for ``POST /api/learn/quiz/{slug}/{assessment_id}/grade``.

    ``answers`` maps ``question_id -> selection`` where a selection is a single
    response id / value (mc / tf / numeric fib) or a list of ids (multiple
    response). Permissive ``extra="allow"`` mirrors ``AskRequest``.
    """

    answers: Dict[str, Any] = {}

    model_config = {"extra": "allow"}


def _quiz_for(slug: str, assessment_id: str):
    """Return ``(assessment, quiz_xml)`` for ``slug``/``assessment_id`` or None.

    Locates every packaged quiz for the course, parses each via the in-tree
    QTI parser, and matches on ``assessment_id``. The original XML is returned
    alongside the parsed object so the grader can resolve multiple-correct keys
    (``grade_submission(..., quiz_xml=...)``). Fail-soft: an unparseable blob is
    skipped, not raised, so one bad quiz never hides the rest.
    """
    from gui.services import quiz_service  # noqa: PLC0415

    for blob in quiz_service.locate_quiz_xml(slug):
        try:
            assessment = quiz_service.parse_quiz(blob)
        except quiz_service.QuizServiceError:
            continue
        if assessment.id == assessment_id:
            return assessment, blob.decode("utf-8", errors="replace")
    return None


@router.get("/quiz/{slug}")
async def list_quizzes(slug: str) -> Any:
    """List a course's quizzes → ``[{assessment_id, title, question_count}]``.

    Learner-open (``/api/learn/*``). Fail-soft: a course with no packaged
    quizzes returns ``[]``; an invalid slug surfaces a typed 422.
    """
    from gui.services import quiz_service  # noqa: PLC0415

    try:
        blobs = quiz_service.locate_quiz_xml(slug)
    except quiz_service.QuizServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected archive/parse failure
        return _error(500, "quiz_list_failed", str(exc))

    out: List[Dict[str, Any]] = []
    for blob in blobs:
        try:
            assessment = quiz_service.parse_quiz(blob)
        except quiz_service.QuizServiceError:
            continue  # skip a malformed quiz, never fail the whole list
        out.append(
            {
                "assessment_id": assessment.id,
                "title": assessment.title,
                "question_count": len(assessment.questions),
            }
        )
    return out


@router.get("/quiz/{slug}/{assessment_id}")
async def get_quiz(slug: str, assessment_id: str) -> Any:
    """Return one quiz's learner JSON — WITHOUT the answer key (§5.1).

    The service's ``to_learner_json`` rebuilds each question from id + text only,
    so no ``correct_answer`` / ``is_correct`` / answer-revealing feedback can leak.
    """
    from gui.services import quiz_service  # noqa: PLC0415

    try:
        found = _quiz_for(slug, assessment_id)
    except quiz_service.QuizServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "quiz_load_failed", str(exc))
    if found is None:
        return _error(404, "quiz_not_found", f"no quiz {assessment_id!r} for course {slug!r}")
    assessment, _xml = found
    return quiz_service.to_learner_json(assessment)


@router.post("/quiz/{slug}/{assessment_id}/grade")
async def grade_quiz(
    slug: str,
    assessment_id: str,
    req: GradeRequest,
    x_learner_id: Optional[str] = Header(default=None),
) -> Any:
    """Server-grade a learner submission → ``{score, max_score, per_item}`` (§5.1).

    Grading is SERVER-SIDE: the answer key is parsed here and never enters the
    learner JSON or the browser. The graded attempt is PERSISTED (W10 §10) when
    the caller presents an ``X-Learner-Id`` owner key — the record keys on that id
    so each learner sees only their own attempts. Grading still returns the same
    payload whether or not the id is present (persistence is best-effort: a write
    failure never blocks the score).
    """
    from gui.services import quiz_service  # noqa: PLC0415

    try:
        found = _quiz_for(slug, assessment_id)
    except quiz_service.QuizServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "quiz_load_failed", str(exc))
    if found is None:
        return _error(404, "quiz_not_found", f"no quiz {assessment_id!r} for course {slug!r}")

    assessment, quiz_xml = found
    answers = dict(req.answers or {})
    try:
        grade = quiz_service.grade_submission(assessment, answers, quiz_xml=quiz_xml)
    except Exception as exc:  # noqa: BLE001 — grading must never 500-leak the key
        return _error(500, "quiz_grade_failed", str(exc))

    # Persist the attempt when an owner key is presented (server-injected ts).
    learner_id = _resolve_learner_id(x_learner_id)
    if learner_id:
        from gui.services import learner_state  # noqa: PLC0415

        try:
            learner_state.record_attempt(slug, assessment_id, learner_id, answers, grade)
        except learner_state.LearnerStateError:
            pass  # a malformed id / slug never blocks returning the score
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass
    return grade


@router.get("/quiz/{slug}/{assessment_id}/attempts")
async def list_attempts(
    slug: str,
    assessment_id: str,
    x_learner_id: Optional[str] = Header(default=None),
) -> Any:
    """Return the CALLER'S OWN graded attempts for one assessment (W10 §10).

    Filtered by the ``X-Learner-Id`` owner key — a learner reads only their own
    attempts (ownership isolation). No id → ``[]`` (an anonymous caller has no
    history). Each attempt carries ``{score, max_score, per_item, answers, ts}``.
    """
    learner_id = _resolve_learner_id(x_learner_id)
    if not learner_id:
        return []
    from gui.services import learner_state  # noqa: PLC0415

    try:
        return learner_state.read_attempts(slug, assessment_id, learner_id)
    except learner_state.LearnerStateError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "attempts_read_failed", str(exc))


# ---------------------------------------------- assignment + discussion (W10 §5.2)
#
# Read-only first cut (no submission, no threading — §10). Assignment + discussion
# XML is located alongside the packaged quizzes (the ``06_assessments/`` sibling +
# the cartridge) and parsed namespace-agnostically. The discussion ``imsdt`` body
# is cartridge-authored HTML, so it is routed through the assessment-path
# ``sanitize_soup(..., strip_form_actions=True)`` hardening (§5.3) before rendering.


def _local_name(tag: str) -> str:
    """Local element name (namespace-agnostic), mirroring quiz_service._local."""
    return tag.rsplit("}", 1)[-1].lower()


def _find_local(root: ET.Element, name: str) -> Optional[ET.Element]:
    """First descendant whose local tag name == ``name`` (namespace-agnostic)."""
    for el in root.iter():
        if _local_name(el.tag) == name:
            return el
    return None


def _text_of(el: Optional[ET.Element]) -> str:
    """Inner XML/text of an element as a string (preserves child HTML markup)."""
    if el is None:
        return ""
    parts: List[str] = []
    if el.text:
        parts.append(el.text)
    for child in list(el):
        parts.append(ET.tostring(child, encoding="unicode"))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _resource_xml_blobs(slug: str) -> List[str]:
    """Every loose ``06_assessments/*.xml`` + cartridge ``*.xml`` member as text.

    Reuses the audited ``quiz_service`` archive-discovery helpers (slug
    validation, cartridge location, in-memory member reads) so assignment +
    discussion locate the SAME way quizzes do — without re-walking the archive or
    editing the quiz service. Fail-soft: an unreadable member is skipped.
    """
    from gui.services import quiz_service  # noqa: PLC0415

    root = quiz_service._libv2_root()
    course_dir = quiz_service._validate_slug(slug, root)

    blobs: List[str] = []
    for rel in quiz_service._ASSESSMENT_ROOTS:
        adir = course_dir / rel
        if not adir.is_dir():
            continue
        for xml_path in sorted(adir.glob("*.xml")):
            try:
                blobs.append(xml_path.read_text(encoding="utf-8"))
            except OSError:
                continue

    cartridge = quiz_service._find_cartridge(course_dir)
    if cartridge is not None:
        import zipfile  # noqa: PLC0415

        try:
            with zipfile.ZipFile(cartridge) as zf:
                for name in sorted(zf.namelist()):
                    if not name.lower().endswith(".xml"):
                        continue
                    try:
                        blobs.append(zf.read(name).decode("utf-8", errors="replace"))
                    except (KeyError, OSError, zipfile.BadZipFile):
                        continue
        except (zipfile.BadZipFile, OSError):
            pass
    return blobs


def _objective_id_of(root: ET.Element) -> str:
    """Best-effort ``objective_id`` from an extensions/metadata field, or ''."""
    for el in root.iter():
        if _local_name(el.tag) in ("objective_id", "objectiveid"):
            return (el.text or "").strip()
        # ``<field label="objective_id">CO-01</field>`` style extension.
        label = el.get("label") or el.get("name") or ""
        if str(label).strip().lower() == "objective_id" and el.text:
            return el.text.strip()
    return ""


def _sanitize_fragment(html_text: str) -> str:
    """Assessment-path sanitize: strip active content + form-submission targets.

    Routes cartridge-authored HTML through the shared
    ``source_page.sanitize_soup`` with ``strip_form_actions=True`` (§5.3) so a
    ``<form action="https://evil/">`` can't exfiltrate learner input. Returns the
    sanitized fragment string.
    """
    from bs4 import BeautifulSoup  # noqa: PLC0415

    from gui.services.source_page import sanitize_soup  # noqa: PLC0415

    soup = BeautifulSoup(html_text, "html.parser")
    sanitize_soup(soup, strip_form_actions=True)
    return str(soup)


def _locate_topic(slug: str, kind: str, target_id: str):
    """Return ``(title, objective_id, body_html)`` for an assignment/discussion.

    ``kind`` is ``"assignment"`` (XSD ``<assignment>``) or ``"discussion"`` (imsdt
    ``<topic>``). Matches on the element's ``identifier``/``ident`` attribute OR,
    when the XML carries no id, falls back to the first such element (a loose
    single-item file). Returns ``None`` when no matching resource exists.
    """
    from gui.services import quiz_service  # noqa: PLC0415

    root_tag = "assignment" if kind == "assignment" else "topic"
    try:
        blobs = _resource_xml_blobs(slug)
    except Exception as exc:  # noqa: BLE001 — map the typed slug error below
        status = getattr(exc, "status", None)
        if status == 422:
            raise quiz_service.QuizServiceError(422, "invalid_course_id", str(exc)) from None
        if status == 404:
            return None
        raise

    fallback: Optional[Tuple[str, str, str]] = None
    for blob in blobs:
        try:
            root = ET.fromstring(blob)
        except ET.ParseError:
            continue
        el = root if _local_name(root.tag) == root_tag else _find_local(root, root_tag)
        if el is None:
            continue
        ident = el.get("identifier") or el.get("ident") or ""
        title = (_text_of(_find_local(el, "title")) or "").strip()
        body = _text_of(_find_local(el, "text"))
        objective_id = _objective_id_of(el)
        record = (title, objective_id, body)
        if ident and ident == target_id:
            return record
        # Fallback ONLY for a loose single-resource file that carried NO
        # identifier at all (the manifest assigns the id externally). An element
        # WITH an id that doesn't match is never served under a different id.
        if not ident and fallback is None:
            fallback = record
    return fallback


@router.get("/assignment/{slug}/{assignment_id}")
async def get_assignment(slug: str, assignment_id: str) -> Any:
    """Render a grounded assignment task (read-only, §5.2) → ``{html, ...}``.

    The task prose is cartridge-authored HTML routed through the assessment-path
    sanitizer (``strip_form_actions=True``). The title + objective id are
    HTML-escaped on render.
    """
    from gui.services import quiz_service  # noqa: PLC0415

    try:
        found = _locate_topic(slug, "assignment", assignment_id)
    except quiz_service.QuizServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "assignment_load_failed", str(exc))
    if found is None:
        return _error(
            404, "assignment_not_found", f"no assignment {assignment_id!r} for course {slug!r}"
        )
    title, objective_id, body = found
    return {
        "assignment_id": assignment_id,
        "title": title,
        "objective_id": objective_id,
        "html": _render_task_fragment("Assignment", title, objective_id, body),
    }


@router.get("/discussion/{slug}/{topic_id}")
async def get_discussion(slug: str, topic_id: str) -> Any:
    """Render a discussion prompt (read-only, §5.2) → ``{html, ...}``.

    The imsdt ``<topic>`` body is cartridge-authored HTML routed through the
    assessment-path sanitizer (``strip_form_actions=True``, §5.3).
    """
    from gui.services import quiz_service  # noqa: PLC0415

    try:
        found = _locate_topic(slug, "discussion", topic_id)
    except quiz_service.QuizServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "discussion_load_failed", str(exc))
    if found is None:
        return _error(
            404, "discussion_not_found", f"no discussion {topic_id!r} for course {slug!r}"
        )
    title, objective_id, body = found
    return {
        "topic_id": topic_id,
        "title": title,
        "objective_id": objective_id,
        "html": _render_task_fragment("Discussion", title, objective_id, body),
    }


def _render_task_fragment(
    label: str, title: str, objective_id: str, body_html: str
) -> str:
    """Server-render a sanitized assignment/discussion fragment.

    Title + objective id are HTML-escaped; the cartridge body is sanitized
    through the assessment-path scrub (active content + off-origin form actions
    stripped). One ``<h2>`` per fragment (slots under the page ``<h1>``).
    """
    heading = escape(title) if title else escape(label)
    obj = (
        '<p class="task-objective">Objective: {}</p>'.format(escape(objective_id))
        if objective_id
        else ""
    )
    safe_body = _sanitize_fragment(body_html) if body_html else ""
    return (
        '<section class="task" data-task-kind="{kind}" aria-labelledby="task-h">'
        '<h2 id="task-h" tabindex="-1">{heading}</h2>'
        "{obj}"
        '<div class="task-body">{body}</div>'
        "</section>"
    ).format(
        kind=escape(label.lower(), quote=True),
        heading=heading,
        obj=obj,
        body=safe_body,
    )


# ----------------------------------------- stateful learner features (W10 §10)
#
# IDENTITY: anonymous per-browser owner key via the ``X-Learner-Id`` header (see
# ``_resolve_learner_id`` + ``gui.services.learner_state`` module docstring). NO
# accounts, NO auth — the id authorizes "owns own records" ONLY. All routes ride
# the existing ``/api/learn/*`` learner-OPEN rule (no operator token). Learner
# prose is sanitized server-side on STORE (defense-in-depth with render escaping).


class DiscussionPostRequest(BaseModel):
    """Body for ``POST /api/learn/discussion/{slug}/{topic_id}/posts``.

    ``body`` is the (HTML-bearing) post text — sanitized server-side on store.
    ``parent_post_id`` (optional) threads a ONE-LEVEL reply.
    """

    body: str = ""
    parent_post_id: Optional[str] = None

    model_config = {"extra": "allow"}


@router.post("/discussion/{slug}/{topic_id}/posts")
async def create_discussion_post(
    slug: str,
    topic_id: str,
    req: DiscussionPostRequest,
    x_learner_id: Optional[str] = Header(default=None),
) -> Any:
    """Persist one discussion post → the stored record (W10 §10 threading).

    The ``body`` is sanitized via ``sanitize_soup(strip_form_actions=True)`` BEFORE
    storage (no stored XSS, no off-origin form exfil). An ``X-Learner-Id`` is
    REQUIRED to author (the post carries its author's id); an empty body after
    sanitize is rejected. ``parent_post_id`` threads a one-level reply.
    """
    learner_id = _resolve_learner_id(x_learner_id)
    if not learner_id:
        return _error(422, "missing_learner_id", "X-Learner-Id header is required to post")
    if not (req.body or "").strip():
        return _error(422, "empty_post", "post body must be non-empty")

    from gui.services import learner_state  # noqa: PLC0415

    try:
        record = learner_state.add_post(
            slug, topic_id, learner_id, req.body, parent_post_id=req.parent_post_id
        )
    except learner_state.LearnerStateError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "post_create_failed", str(exc))
    return record


@router.get("/discussion/{slug}/{topic_id}/posts")
async def list_discussion_posts(slug: str, topic_id: str) -> Any:
    """Return the full thread for a topic (all learners; world-readable, §10).

    A discussion is a shared thread: every learner's posts are returned (only the
    author id is attached, never any private record). Body text is the
    server-sanitized ``body_sanitized`` (the front-end escapes on render too).
    """
    from gui.services import learner_state  # noqa: PLC0415

    try:
        return learner_state.read_posts(slug, topic_id)
    except learner_state.LearnerStateError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "posts_read_failed", str(exc))


@router.post("/assignment/{slug}/{assignment_id}/submit")
async def submit_assignment(
    slug: str,
    assignment_id: str,
    request: Request,
    x_learner_id: Optional[str] = Header(default=None),
) -> Any:
    """Persist a learner's assignment submission (text + optional file) — W10 §10.

    Accepts EITHER a JSON body ``{text, ...}`` OR a ``multipart/form-data`` body
    carrying ``text`` + an optional single ``file`` upload. The text is sanitized
    on store; the file is validated (extension allowlist ``.pdf/.txt/.md/.docx``,
    5 MB cap, traversal-safe opaque blob name) and stored as an opaque blob under
    the caller's own ``submissions/`` dir. An ``X-Learner-Id`` owner key is
    REQUIRED (the submission keys on it for ownership isolation). Text submission
    is required; the file is optional. Returns the stored record (no file bytes).
    """
    learner_id = _resolve_learner_id(x_learner_id)
    if not learner_id:
        return _error(422, "missing_learner_id", "X-Learner-Id header is required to submit")

    from gui.services import learner_state  # noqa: PLC0415

    text = ""
    file_bytes: Optional[bytes] = None
    original_filename: Optional[str] = None

    content_type = (request.headers.get("content-type") or "").lower()
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            text = str(form.get("text") or "")
            upload = form.get("file")
            if upload is not None and hasattr(upload, "filename"):
                original_filename = upload.filename or ""
                file_bytes = await upload.read()
        else:
            body = await request.json()
            if not isinstance(body, dict):
                return _error(422, "invalid_submission", "expected a JSON object body")
            text = str(body.get("text") or "")
    except Exception as exc:  # noqa: BLE001 — malformed body
        return _error(422, "invalid_submission", f"could not parse submission: {exc}")

    if not text.strip() and file_bytes is None:
        return _error(422, "empty_submission", "a text submission is required")

    try:
        record = learner_state.save_submission(
            slug,
            assignment_id,
            learner_id,
            text,
            file_bytes=file_bytes,
            original_filename=original_filename,
        )
    except learner_state.LearnerStateError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "submission_failed", str(exc))
    return record


@router.get("/assignment/{slug}/{assignment_id}/submission")
async def get_submission(
    slug: str,
    assignment_id: str,
    x_learner_id: Optional[str] = Header(default=None),
) -> Any:
    """Return the CALLER'S OWN assignment submission, or 404 (W10 §10).

    Ownership isolation: keyed on the ``X-Learner-Id`` owner key — a learner reads
    only their own submission. No id → 422; no submission → 404.
    """
    learner_id = _resolve_learner_id(x_learner_id)
    if not learner_id:
        return _error(422, "missing_learner_id", "X-Learner-Id header is required")

    from gui.services import learner_state  # noqa: PLC0415

    try:
        record = learner_state.read_submission(slug, assignment_id, learner_id)
    except learner_state.LearnerStateError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "submission_read_failed", str(exc))
    if record is None:
        return _error(
            404, "submission_not_found", f"no submission for {assignment_id!r} by this learner"
        )
    return record


__all__ = ["router"]
