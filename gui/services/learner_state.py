"""Shared learner-state persistence layer — W10 §10 STATEFUL learner features.

This module is the single, audited persistence core behind the three stateful
learner surfaces the W10 spec deferred (spec
``plans/finegrain/w10-assessments-qti-discussions-2026-06.md`` §10):

1. **quiz ATTEMPT/GRADE persistence** — a graded attempt is recorded so a learner
   can see their own history (`record_attempt` / `read_attempts`).
2. **discussion THREADING** — append-only posts with one-level replies
   (`add_post` / `read_posts`).
3. **assignment SUBMISSION** — text + an optional single safe file blob
   (`save_submission` / `read_submission`, file handling in `save_blob`).

═══ PINNED DESIGN (the documented defaults) ═══════════════════════════════════

* **IDENTITY — anonymous per-browser learner id.** There are NO real accounts and
  NO auth (this preserves the learner-OPEN posture of spec §1.5). The front-end
  mints a uuid in ``localStorage`` and sends it as an ``X-Learner-Id`` request
  header. The server treats it as an *opaque owner key* only — it is NEVER trusted
  for authorization beyond "a learner reads/writes only their own records". A
  learner who presents learner B's id could read B's records, exactly as a learner
  who knows B's localStorage uuid could; this is the intended trust level for an
  open, account-less learner surface (it is privacy-by-obscure-id, not security —
  documented honestly). Discussion posts are world-readable *within a course*;
  only the author's id is attached.

* **STORAGE — file-based under the relocatable state dir.** Everything lands under
  ``<state>/gui/learn/<course_slug>/`` where ``<state>`` is ``lib.paths.STATE_PATH``
  (which honors ``ED4ALL_HOME`` / ``ED4ALL_STATE_RUNS_DIR``; resolved at call time
  via ``gui.shared_state._state_root`` so tests redirect into a ``tmp_path``).
  Layout::

      <state>/gui/learn/<slug>/
      ├── attempts.jsonl          # append-only: one line per graded attempt
      ├── discussion/<topic_id>.jsonl   # append-only: one line per post
      └── submissions/<assignment_id>/<learner_id>/
          ├── submission.json     # the text submission record (atomic overwrite)
          └── <opaque-blob-name>  # the (optional) uploaded file, stored verbatim

  Append-only JSONL is used for attempts + posts (history); a submission is the
  learner's single current submission so it is an atomic overwrite. Appends are
  serialized per-target by a process+thread lock (``_locked``) so two concurrent
  appends can't interleave a half-written line, and the directory-level lock makes
  the open-append-flush sequence atomic within this process.

* **SURFACE — learner-OPEN only.** This layer is consumed exclusively by
  ``/api/learn/*`` routes. There is NO instructor-review / grade-export surface
  here (that would be operator-gated per §1.5 and is out of scope).

Security posture (hard contracts, defense-in-depth with the route layer):

* The course slug is validated against ``_SLUG_RE`` (and rejected on any path
  separator) before it is used to build a path — a traversal slug never escapes
  the learn root. The learner id, topic id, and assignment id are likewise
  path-separator-guarded via ``_safe_component``.
* All learner-submitted prose (discussion bodies, assignment text) is sanitized
  HERE on STORE via ``gui.services.source_page.sanitize_soup(strip_form_actions=True)``
  so no active content / off-origin form action is ever persisted (the route layer
  ALSO escapes on render — belt and suspenders).
* File uploads enforce a strict extension allowlist, a byte-size cap, and a
  path-traversal-safe opaque blob name; the blob is stored verbatim and never
  executed or served as HTML.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui import shared_state

# --------------------------------------------------------------------------- #
# Identity / path-safety primitives
# --------------------------------------------------------------------------- #

# Course-slug shape — identical to imscc_service._SLUG_RE / source_page._SLUG_RE.
_SLUG_RE = re.compile(r"^[a-z0-9][a-zA-Z0-9._-]*$")

# Opaque-component shape for learner id / topic id / assignment id. Permissive
# (a localStorage uuid, a manifest topic ident) but path-separator-free.
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")

# File-upload safety (§ assignment submission).
ALLOWED_UPLOAD_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".docx"})
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# A per-target lock registry so concurrent appends to the SAME file serialize.
_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class LearnerStateError(Exception):
    """Raised on an invalid slug / id / rejected upload.

    ``status`` + ``code`` mirror the ``QuizServiceError`` (status, code, detail)
    contract so the route layer surfaces a clean typed HTTP error.
    """

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def _now_iso() -> str:
    """UTC ISO-8601 timestamp (server-injected; never a client-supplied ts)."""
    return datetime.now(timezone.utc).isoformat()


def _validate_slug(slug: str) -> str:
    """Return ``slug`` if it matches the course-slug shape, else raise 422.

    Defence-in-depth: rejects any path separator / traversal so a hostile slug
    never escapes the learn root.
    """
    if not isinstance(slug, str) or not slug or not _SLUG_RE.match(slug):
        raise LearnerStateError(422, "invalid_course_id", f"invalid course slug: {slug!r}")
    if "/" in slug or os.sep in slug or ".." in slug:
        raise LearnerStateError(422, "invalid_course_id", f"slug has a separator: {slug!r}")
    return slug


def _safe_component(value: str, label: str) -> str:
    """Validate an opaque path component (learner/topic/assignment id), else 422."""
    if not isinstance(value, str) or not _COMPONENT_RE.match(value):
        raise LearnerStateError(422, f"invalid_{label}", f"invalid {label}: {value!r}")
    if "/" in value or os.sep in value or ".." in value:
        raise LearnerStateError(422, f"invalid_{label}", f"{label} has a separator: {value!r}")
    return value


# --------------------------------------------------------------------------- #
# Path resolution (under the relocatable state dir)
# --------------------------------------------------------------------------- #


def _learn_root() -> Path:
    """Return ``<state>/gui/learn/`` (created lazily).

    Resolved through ``shared_state.gui_state_dir()`` which honors
    ``ED4ALL_STATE_RUNS_DIR`` / ``ED4ALL_HOME`` at call time, so tests redirect
    the whole store into a ``tmp_path`` and a relocated deploy lands under
    ``ED4ALL_HOME``.
    """
    path = shared_state.gui_state_dir() / "learn"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _course_dir(slug: str) -> Path:
    """Return ``<state>/gui/learn/<slug>/`` for a validated slug (created lazily)."""
    path = _learn_root() / _validate_slug(slug)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lock_for(key: str) -> threading.Lock:
    """Return a process-wide lock for ``key`` (one lock per append target)."""
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record as a line under a per-file lock (corruption-safe).

    The lock serializes the open-append-flush so two concurrent appends never
    interleave a half-written line. ``flush()`` + ``fsync`` keep the line durable
    before the lock is released, so a reader after the append sees a complete
    record. ``json.dumps`` never emits a newline (separators are comma/space), so
    one record == one line by construction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str)
    with _lock_for(str(path)):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Return every JSON record from ``path`` (skip malformed/blank lines)."""
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with _lock_for(str(path)):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _sanitize_text(raw: str) -> str:
    """Sanitize learner-submitted HTML/prose via the shared assessment-path scrub.

    Routes through ``source_page.sanitize_soup(strip_form_actions=True)`` so a
    ``<script>`` / ``on*`` handler / ``<form action="https://evil/">`` is stripped
    BEFORE the content is ever persisted (no stored XSS, no off-origin exfil). The
    import is lazy so this module imports without bs4 on a non-serving path.
    """
    if not raw:
        return ""
    from bs4 import BeautifulSoup  # noqa: PLC0415

    from gui.services.source_page import sanitize_soup  # noqa: PLC0415

    soup = BeautifulSoup(raw, "html.parser")
    sanitize_soup(soup, strip_form_actions=True)
    return str(soup)


# --------------------------------------------------------------------------- #
# (1) Quiz attempt persistence
# --------------------------------------------------------------------------- #


def record_attempt(
    slug: str,
    assessment_id: str,
    learner_id: str,
    answers: Dict[str, Any],
    grade: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist one graded attempt; return the stored record (with server ``ts``).

    ``grade`` is the ``quiz_service.grade_submission`` result
    (``{score, max_score, per_item}``). The ``ts`` is injected by the SERVER here
    (``datetime`` — never a client-supplied timestamp). Append-only; a learner's
    attempts accrue in order.
    """
    slug = _validate_slug(slug)
    assessment_id = _safe_component(assessment_id, "assessment_id")
    learner_id = _safe_component(learner_id, "learner_id")

    record = {
        "learner_id": learner_id,
        "assessment_id": assessment_id,
        "answers": dict(answers or {}),
        "score": grade.get("score"),
        "max_score": grade.get("max_score"),
        "per_item": grade.get("per_item", []),
        "ts": _now_iso(),
    }
    _append_jsonl(_course_dir(slug) / "attempts.jsonl", record)
    return record


def read_attempts(
    slug: str, assessment_id: str, learner_id: str
) -> List[Dict[str, Any]]:
    """Return the CALLER'S OWN attempts for one assessment (oldest first).

    Filtered by BOTH ``assessment_id`` AND ``learner_id`` — a learner sees only
    their own attempts (ownership isolation; the X-Learner-Id is an owner key).
    """
    slug = _validate_slug(slug)
    assessment_id = _safe_component(assessment_id, "assessment_id")
    learner_id = _safe_component(learner_id, "learner_id")

    rows = _read_jsonl(_course_dir(slug) / "attempts.jsonl")
    return [
        r
        for r in rows
        if r.get("assessment_id") == assessment_id
        and r.get("learner_id") == learner_id
    ]


# --------------------------------------------------------------------------- #
# (2) Discussion threading
# --------------------------------------------------------------------------- #


def _topic_path(slug: str, topic_id: str) -> Path:
    return _course_dir(slug) / "discussion" / f"{_safe_component(topic_id, 'topic_id')}.jsonl"


def add_post(
    slug: str,
    topic_id: str,
    learner_id: str,
    body: str,
    parent_post_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist one discussion post; return the stored record.

    ``body`` is sanitized HERE via ``sanitize_soup(strip_form_actions=True)`` (no
    stored XSS). ``parent_post_id`` threads a ONE-LEVEL reply: a reply to a reply
    is flattened to a reply to the top-level root (so the thread is at most one
    deep), and a ``parent_post_id`` that doesn't resolve to an existing top-level
    post in this topic is dropped (the post becomes top-level). The ``post_id`` +
    ``ts`` are server-minted.
    """
    slug = _validate_slug(slug)
    topic_id = _safe_component(topic_id, "topic_id")
    learner_id = _safe_component(learner_id, "learner_id")

    resolved_parent: Optional[str] = None
    if parent_post_id:
        parent_post_id = _safe_component(parent_post_id, "post_id")
        for existing in _read_jsonl(_topic_path(slug, topic_id)):
            if existing.get("post_id") == parent_post_id:
                # One-level only: a reply to a reply re-parents to the root.
                resolved_parent = existing.get("parent_post_id") or parent_post_id
                break

    record = {
        "post_id": shared_state.new_run_id(prefix="POST"),
        "learner_id": learner_id,
        "topic_id": topic_id,
        "body_sanitized": _sanitize_text(body or ""),
        "parent_post_id": resolved_parent,
        "ts": _now_iso(),
    }
    _append_jsonl(_topic_path(slug, topic_id), record)
    return record


def read_posts(slug: str, topic_id: str) -> List[Dict[str, Any]]:
    """Return the full thread for ``topic_id`` (all learners; oldest first).

    World-readable within the course (a discussion is a shared thread). Only the
    author's ``learner_id`` is attached to each post — there is no owner filter
    here by design.
    """
    slug = _validate_slug(slug)
    topic_id = _safe_component(topic_id, "topic_id")
    return _read_jsonl(_topic_path(slug, topic_id))


# --------------------------------------------------------------------------- #
# (3) Assignment submission
# --------------------------------------------------------------------------- #


def _submission_dir(slug: str, assignment_id: str, learner_id: str) -> Path:
    path = (
        _course_dir(slug)
        / "submissions"
        / _safe_component(assignment_id, "assignment_id")
        / _safe_component(learner_id, "learner_id")
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_upload(filename: str, size: int) -> str:
    """Validate an upload's filename + size; return the SAFE opaque blob name.

    Hard controls (spec §3 file-upload safety):
      * **extension allowlist** — only ``.pdf/.txt/.md/.docx`` (case-insensitive).
      * **size cap** — ``MAX_UPLOAD_BYTES`` (5 MB).
      * **no path traversal** — the stored blob name is rebuilt from the basename
        only; any directory component / ``..`` in the supplied filename is
        discarded. The blob is stored verbatim (an opaque blob), never executed,
        never served as HTML.

    Raises ``LearnerStateError`` (413 too large, 415 bad type, 422 bad name).
    """
    if not isinstance(filename, str) or not filename.strip():
        raise LearnerStateError(422, "invalid_filename", "empty upload filename")
    # Strip every directory component — defeat ``../`` and absolute paths. The
    # basename is taken from BOTH separators so a Windows-style ``..\x`` can't slip
    # a separator past a POSIX basename.
    base = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not base or base in (".", "..") or base.startswith("."):
        raise LearnerStateError(422, "invalid_filename", f"unsafe filename: {filename!r}")
    ext = Path(base).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise LearnerStateError(
            415,
            "unsupported_file_type",
            f"extension {ext!r} not in allowlist {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )
    if size is not None and size > MAX_UPLOAD_BYTES:
        raise LearnerStateError(
            413, "file_too_large", f"upload {size} bytes exceeds cap {MAX_UPLOAD_BYTES}"
        )
    # Opaque, traversal-safe stored name: keep only the validated extension and a
    # fixed safe base. The original (untrusted) name is recorded in the JSON
    # record for display, but never used as a path.
    return f"upload{ext}"


def save_submission(
    slug: str,
    assignment_id: str,
    learner_id: str,
    text: str,
    file_bytes: Optional[bytes] = None,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a learner's assignment submission (text + optional single file).

    ``text`` is sanitized HERE (no stored XSS). The optional file is validated
    (``validate_upload``) and stored as an opaque blob under
    ``submissions/<assignment_id>/<learner_id>/``; the blob is never executed nor
    served as HTML. A submission is the learner's single CURRENT submission, so
    the record is an atomic overwrite (not append-only). Returns the stored
    record (without the file bytes).
    """
    slug = _validate_slug(slug)
    assignment_id = _safe_component(assignment_id, "assignment_id")
    learner_id = _safe_component(learner_id, "learner_id")

    sdir = _submission_dir(slug, assignment_id, learner_id)

    file_meta: Optional[Dict[str, Any]] = None
    if file_bytes is not None and len(file_bytes) > 0:
        blob_name = validate_upload(original_filename or "", len(file_bytes))
        if len(file_bytes) > MAX_UPLOAD_BYTES:  # defence-in-depth re-check on bytes
            raise LearnerStateError(
                413, "file_too_large", f"upload {len(file_bytes)} bytes exceeds cap"
            )
        blob_path = sdir / blob_name
        # Atomic write so a reader never sees a partial blob; 0600 (owner-only).
        tmp = sdir / f".{blob_name}.tmp"
        with _lock_for(str(blob_path)):
            tmp.write_bytes(file_bytes)
            os.chmod(tmp, 0o600)
            os.replace(tmp, blob_path)
        file_meta = {
            "stored_name": blob_name,
            "original_filename": str(original_filename or ""),
            "size": len(file_bytes),
        }

    record = {
        "learner_id": learner_id,
        "assignment_id": assignment_id,
        "text_sanitized": _sanitize_text(text or ""),
        "file": file_meta,
        "ts": _now_iso(),
    }
    shared_state._atomic_write_json(sdir / "submission.json", record)
    return record


def read_submission(
    slug: str, assignment_id: str, learner_id: str
) -> Optional[Dict[str, Any]]:
    """Return the CALLER'S OWN submission for one assignment, or ``None``.

    Ownership isolation: the path is keyed on ``learner_id`` so a learner can only
    read their own submission record — there is no cross-learner read path.
    """
    slug = _validate_slug(slug)
    assignment_id = _safe_component(assignment_id, "assignment_id")
    learner_id = _safe_component(learner_id, "learner_id")

    path = _submission_dir(slug, assignment_id, learner_id) / "submission.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_submission_blob(
    slug: str, assignment_id: str, learner_id: str
) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    """Return ``(bytes, file_meta)`` for the caller's uploaded blob, or ``None``.

    Used by a download route to stream the learner their own opaque blob. Returns
    ``None`` when there is no submission or no attached file. The blob is read from
    the validated ``stored_name`` only (never a client-supplied path).
    """
    record = read_submission(slug, assignment_id, learner_id)
    if not record or not record.get("file"):
        return None
    meta = record["file"]
    stored = meta.get("stored_name")
    if not stored or "/" in stored or os.sep in stored or ".." in stored:
        return None
    sdir = _submission_dir(slug, assignment_id, learner_id)
    blob_path = sdir / stored
    if not blob_path.is_file():
        return None
    try:
        return blob_path.read_bytes(), meta
    except OSError:
        return None


__all__ = [
    "LearnerStateError",
    "ALLOWED_UPLOAD_EXTENSIONS",
    "MAX_UPLOAD_BYTES",
    "record_attempt",
    "read_attempts",
    "add_post",
    "read_posts",
    "validate_upload",
    "save_submission",
    "read_submission",
    "read_submission_blob",
]
