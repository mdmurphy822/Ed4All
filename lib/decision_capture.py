#!/usr/bin/env python3
"""
Claude Decision Capture Utility for Ed4All

Captures all Claude decision points during:
- SemantiK PDF conversion
- CourseForge course generation
- Trainforge assessment generation

Logs prompts, responses, and decisions for training data collection.

Phase 0 Hardening Enhancements:
- event_id and seq for monotonic ordering
- task_id for cross-linking to orchestrator tasks
- is_default flag for non-decision capture
- Enhanced inputs[] with hash_algorithm
- New outputs[] array as artifact pointers
- Integration with run context for hardened mode

Adapted from INTEGRATOR CURRICULUM decision_capture.py
"""

import atexit
import hashlib
import json
import logging
import os
import secrets
import threading
import weakref
from dataclasses import asdict, dataclass, field, is_dataclass

# Worker W3: advisory file locking around the legacy decision-capture write
# path. fcntl is POSIX-only; Windows / non-POSIX environments degrade to the
# pre-W3 unlocked behavior. flock is also unreliable on WSL2 DrvFS (`/mnt/c/`)
# and some NFS mounts; the runtime OSError fallback inside _write_to_streams
# handles that case.
try:
    import fcntl as _fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - non-POSIX (Windows) fallback
    _fcntl = None  # type: ignore[assignment]
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import (
    MIN_DECISIONS_PER_PHASE,
    OPERATION_MAP,
    VALIDATE_DECISIONS,
)
from .libv2_storage import LibV2Storage
from .paths import TRAINING_DIR as LEGACY_TRAINING_DIR  # noqa: F401  back-compat re-export
from .paths import get_training_captures_dir


def _coerce_record_field(value: Any) -> Any:
    """Coerce a single decision-record field to its serialisable form.

    Most call sites pass plain ``Dict[str, Any]`` for ``ml_features`` / ``outcome``
    so they can ship per-event fields (``tier``, ``validator``, …) that the
    canonical ``MLFeatures`` dataclass doesn't carry. ``asdict()`` raises on
    those, the defensive try/except at the call site swallows the raise, and
    the decision event silently loses its ``ml_features`` payload. Branch on
    dataclass-instance vs already-coerced shape so both paths preserve the
    payload.
    """
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _coerce_record_field_list(values: Optional[List[Any]]) -> List[Any]:
    """List counterpart of :func:`_coerce_record_field`."""
    if not values:
        return []
    return [_coerce_record_field(item) for item in values]
from .quality import assess_decision_quality

# Phase 0 hardening components are optional independently.  Do not let an
# absent RunManager disable the always-available sequence allocator.
try:
    from .provenance import InputRef as ProvenanceInputRef  # noqa: F401
    from .provenance import OutputRef, create_input_ref, create_output_ref  # noqa: F401
except ImportError:
    pass

try:
    from .run_manager import HARDENED_MODE, get_current_run
    HARDENING_AVAILABLE = True
except ImportError:
    HARDENING_AVAILABLE = False
    HARDENED_MODE = False

    def get_current_run():
        return None

from .sequence_manager import generate_event_id, get_sequence_for_context

# Phase 0.5: WriteFacade for centralized write discipline
try:
    from .path_constants import is_write_facade_enforced
    from .write_facade import WriteFacade, WriteResult  # noqa: F401
    WRITE_FACADE_AVAILABLE = True
except ImportError:
    WRITE_FACADE_AVAILABLE = False

    def is_write_facade_enforced():
        return False

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    """Parse-with-fallback truthy env resolver (``{1,true,yes,on}``)."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


# Builder-2 (decision-capture buffering). ``ED4ALL_CAPTURE_BUFFER`` (default
# OFF) coalesces the per-decision ``write + flush + os.fsync`` on all three
# JSONL mirrors (canonical LibV2 store + legacy training-captures mirror +
# run-specific stream) into a single batched write every
# ``ED4ALL_CAPTURE_BUFFER_ROWS`` rows (default 50). PROFILE: each
# ``log_decision`` currently fsyncs 3 open handles, so a metadata-only gate
# emitting ~424 decisions pays ~424*3 disk syncs (~5.8s/gate); batching cuts
# the fsync count by ``ED4ALL_CAPTURE_BUFFER_ROWS``x.
#
# CRASH-LOSS TRADE: a hard crash / SIGKILL between flushes loses at most the
# last ``ED4ALL_CAPTURE_BUFFER_ROWS`` buffered rows. Those rows are ADVISORY
# TELEMETRY (decision captures for offline training-data mining) — NOT gate
# verdicts, which live in the gate-result stream elsewhere — so the loss
# window is telemetry-only. The buffer is drained on ``flush()``, on
# ``close()`` / ``__del__`` / ``save()``, and via an ``atexit`` hook, so a
# clean or signal-handled shutdown loses nothing.
#
# Default OFF ⇒ byte-identical per-call write path (each decision fsynced
# immediately, exactly as before this flag landed). Garbage / falsey ⇒ off.
_DEFAULT_CAPTURE_BUFFER_ROWS = 50


def _resolve_capture_buffer_rows() -> int:
    """Resolve ``ED4ALL_CAPTURE_BUFFER_ROWS`` (default 50; garbage/<=0 → 50)."""
    raw = os.environ.get("ED4ALL_CAPTURE_BUFFER_ROWS", "").strip()
    if not raw:
        return _DEFAULT_CAPTURE_BUFFER_ROWS
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CAPTURE_BUFFER_ROWS
    return n if n > 0 else _DEFAULT_CAPTURE_BUFFER_ROWS


# ``normalize_course_code`` is the shared course-code coercer. It lives in
# this shared decision-capture module — the canonical home — so the
# orchestrator-level ``PipelineOrchestrator._get_executor`` can use the same
# normalisation without importing from a sibling MCP tool module (avoiding a
# dependency inversion between ``lib/`` and ``MCP/tools/``). Import this name
# from ``lib.decision_capture``.
#
# Canonical pattern: ``^[A-Z]{2,8}_[0-9]{3}$`` (2-8 uppercase letters,
# underscore, 3 digits). Raw course-code strings without the required
# prefix_NNN format (e.g. a product name like ``"Ed4All"`` or a
# long slug-style filename) don't match out of the box, so captures
# previously carried a ``course_id`` validation issue (observed at
# ~50% of records on a recent run). Normalisation strategy:
#
# 1. Uppercase + replace any non-alphanumeric with underscore.
# 2. Strip leading/trailing underscores + collapse repeats.
# 3. If the result already matches the pattern, return as-is.
# 4. Otherwise, split on ``_`` and use the first purely-alphabetic
#    chunk (truncated to 8 chars) as the prefix. If no alphabetic
#    chunk exists, use ``"PDF"`` as the fallback prefix.
# 5. Derive a deterministic 3-digit numeric suffix from the full raw
#    name via SHA-256 modulo 1000 so the same PDF always produces the
#    same course code.
import re as _re_norm

_COURSE_CODE_PATTERN = _re_norm.compile(r"^[A-Z]{2,8}_[0-9]{3}$")


def normalize_course_code(raw: str) -> str:
    """Coerce a raw course code / PDF name into ``^[A-Z]{2,8}_[0-9]{3}$``.

    Examples
    --------
    >>> normalize_course_code("MTH_101")
    'MTH_101'
    >>> normalize_course_code("Ed4All")  # doctest: +ELLIPSIS
    'ED_...'
    """
    raw = (raw or "").strip()
    if not raw:
        raw = "unknown"

    uppered = _re_norm.sub(r"[^A-Za-z0-9]+", "_", raw).upper().strip("_")
    uppered = _re_norm.sub(r"_+", "_", uppered)

    if _COURSE_CODE_PATTERN.match(uppered):
        return uppered

    chunks = [c for c in uppered.split("_") if c]
    prefix = ""
    for chunk in chunks:
        alpha_only = _re_norm.sub(r"[^A-Z]", "", chunk)
        if len(alpha_only) >= 2:
            prefix = alpha_only[:8]
            break
    if not prefix:
        prefix = "PDF"
    if len(prefix) < 2:
        prefix = (prefix + "PDF")[:2]
    prefix = prefix[:8]

    suffix_int = int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % 1000
    suffix = f"{suffix_int:03d}"
    candidate = f"{prefix}_{suffix}"
    if not _COURSE_CODE_PATTERN.match(candidate):
        candidate = f"PDF_{suffix}"
    return candidate


# ADR-001 Contract 3 + REC-CTR-04 (Worker G wave 1.1): decision-type registry.
#
# Source of truth is ``schemas/events/decision_event.schema.json``. The
# ``ALLOWED_DECISION_TYPES`` tuple is loaded at module-import time from the
# schema's ``properties.decision_type.enum`` via :func:`lib.validation.load_schema`.
#
# Convention: add a new type to the schema in the same PR that first uses it
# in production. New types are ``snake_case`` and tool-prefixed when ambiguous.
#
# The registry is advisory by default (warn-only on unknown types), preserving
# backward compat for legacy callers across the tree that may still emit
# free-string types not yet catalogued in the schema. Fail-closed enforcement
# is opt-in via the ``DECISION_VALIDATION_STRICT=true`` environment variable
# (see :meth:`DecisionCapture._validate_record`).
#
# On schema-load failure we fall back to a minimal tuple so this module still
# imports cleanly in environments where the schema file is not reachable
# (e.g., packaging edge cases, minimal test harnesses).
try:
    from .validation import load_schema as _load_schema
    _decision_schema = _load_schema("decision_event")
    ALLOWED_DECISION_TYPES: tuple = tuple(
        _decision_schema["properties"]["decision_type"]["enum"]
    )
except Exception as _e:  # pragma: no cover - defensive fallback
    logger.warning(
        "Failed to load decision_event schema for ALLOWED_DECISION_TYPES: %s; "
        "falling back to minimal tuple",
        _e,
    )
    ALLOWED_DECISION_TYPES: tuple = (
        "instruction_pair_synthesis",
        "preference_pair_generation",
        "typed_edge_inference",
    )


@dataclass
class MLFeatures:
    """Categorical fields for ML training."""
    pedagogy_pattern: str = ""  # "problem_based_intro", "worked_examples", etc.
    engagement_patterns: List[str] = field(default_factory=list)
    cognitive_load_strategy: List[str] = field(default_factory=list)
    bloom_levels: List[str] = field(default_factory=list)
    udl_principles: List[str] = field(default_factory=list)
    component_types: List[str] = field(default_factory=list)


@dataclass
class InputRef:
    """Reference to input sources used for a decision."""
    source_type: str  # "textbook", "existing_imscc", "web_search", "prompt_template", "pdf", "assessment_bank"
    path_or_id: str   # File path, URL, or template ID
    content_hash: str = ""  # SHA256 of content (first 12 chars)
    hash_algorithm: str = "sha256"  # Phase 0: Algorithm used for hashing
    excerpt_range: str = ""  # "lines:100-200" or "pages:15-20"
    size_bytes: int = 0  # Phase 0: Size of content
    byte_range: Optional[Dict[str, int]] = None  # Phase 0: {"start": 0, "end": 100}


@dataclass
class OutputArtifact:
    """Reference to output artifact produced by a decision (Phase 0 Hardening)."""
    artifact_type: str  # "html", "imscc", "assessment", "chunk", etc.
    path: str  # File path relative to run artifacts
    content_hash: str = ""  # Hash of content
    hash_algorithm: str = "sha256"
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeSignals:
    """Outcome tracking for preference/reward training."""
    accepted: bool = True
    revision_count: int = 0
    edit_distance: str = "none"  # "none", "low", "medium", "high"
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


class DecisionCapture:
    """Captures Claude decisions and reasoning for training data."""

    def __init__(
        self,
        course_code: str,
        phase: str,
        tool: str = "courseforge",
        streaming: bool = True,
        task_id: Optional[str] = None,  # Phase 0: Cross-link to orchestrator task
    ):
        """
        Initialize decision capture.

        Args:
            course_code: Course code (e.g., "MTH_101")
            phase: Pipeline phase (e.g., "input-research", "content-generator")
            tool: "semantik", "courseforge", or "trainforge"
            streaming: If True, write decisions immediately to disk (crash-safe)
            task_id: Phase 0 - Orchestrator task ID for cross-linking
        """
        self.course_code = course_code
        self.phase = phase
        self.tool = tool
        # Worker W6: 1-second granularity collides under multi-worker fanout
        # (SemantiK per-PDF max_concurrent=4, assessment_generation max_concurrent=5,
        # etc.). Append PID + 6-char random hex suffix so two captures
        # initialized in the same wall-second from parallel workers (or even
        # the same process) get distinct session_ids, JSONL paths, and run_id
        # fallbacks. Timestamp prefix `%Y%m%d_%H%M%S` (chars 0..15) preserved
        # so existing glob patterns `decisions_*.jsonl` still match.
        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{os.getpid()}_{secrets.token_hex(3)}"
        )
        self.streaming_mode = streaming
        self.task_id = task_id  # Phase 0: Cross-link to orchestrator

        # Phase 0 Hardening: Check for active run context
        self._run_context = get_current_run() if HARDENING_AVAILABLE else None

        # Stable IDs for ML training
        if self._run_context:
            # Use run_id from active run context
            self.run_id = self._run_context.run_id
        else:
            self.run_id = (
                os.environ.get("ED4ALL_RUN_ID")
                or os.environ.get("RUN_ID")
                or f"{tool}_{course_code}_{self.session_id}"
            )

        self.course_id = course_code.replace(' ', '_').upper()  # Normalized: "MTH_101"
        self.module_id: Optional[str] = None  # Set per-module via set_module_context()
        self.artifact_id: Optional[str] = None  # Hash of produced content

        # Use LibV2 storage for training captures (primary location).
        #
        # ROOT FIX: the storage handle is built with ``auto_create=False`` (see
        # ``_build_storage``) so logging a decision NEVER creates an empty
        # ``LibV2/courses/<slug>/`` skeleton — the recurring split-brain "empty
        # twin" collateral. The capture write location is unchanged: the next
        # line's ``get_training_capture_path`` mkdir's its own dir under
        # ``self.training_path`` (``catalog/<course_id>/training/…``), which
        # ``LibV2Storage.__init__`` sets independent of ``auto_create``.
        self._storage = self._build_storage(course_code, tool, phase)
        self.output_dir = self._storage.get_training_capture_path(tool, phase)

        # Legacy training-captures directory (secondary location per CLAUDE.md spec).
        # ``phase=None`` is permitted by the canonical decision-event schema —
        # route to ``phase_unknown`` so tool-level captures (e.g. orchestrator
        # phase_start emits before a phase has been selected) don't crash.
        #
        # The legacy mirror root is resolved at construction time. NOTE:
        # ``ED4ALL_LIBV2_ROOT`` does NOT govern ``runtime/training-captures/`` — the
        # mirror lives at the project root, not under LibV2, so it needs its
        # own override. The repo-root conftest autouse isolation fixture sets
        # ``ED4ALL_TRAINING_CAPTURES_DIR`` to tmp so a pytest run that exercises
        # any LLM call path no longer grows the real ``runtime/training-captures/`` tree
        # by thousands of files.
        #
        # Resolution precedence:
        #   1. ``ED4ALL_TRAINING_CAPTURES_DIR`` env var (autouse isolation;
        #      env vars also propagate to subprocess-spawned writers).
        #   2. ``get_training_captures_dir()`` — reads ``lib.paths.TRAINING_DIR``
        #      at call time, so a session-level monkeypatch of the default
        #      constant is honored.
        #
        # The module-level ``LEGACY_TRAINING_DIR`` global is retained as a
        # back-compat re-export only; it is deliberately NOT consulted here.
        # (A "was-it-patched?" heuristic was tried and reverted: a test that
        # ``importlib.reload``-s this module INSIDE an active
        # ``mock.patch('...LEGACY_TRAINING_DIR')`` context — see
        # ``lib/tests/test_decision_capture_logging.py`` — leaves the global
        # pointing at the pre-reload REAL path after the patch's restore,
        # which the heuristic misread as a deliberate redirect to the real
        # tree.)
        env_captures = os.environ.get("ED4ALL_TRAINING_CAPTURES_DIR", "").strip()
        if env_captures:
            legacy_training_dir = Path(env_captures)
        else:
            legacy_training_dir = get_training_captures_dir()
        normalized_phase = phase.replace("_", "-") if phase else "unknown"
        self.legacy_output_dir = legacy_training_dir / tool / course_code / f"phase_{normalized_phase}"
        self.legacy_output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 0 Hardening: Also write to run-specific decisions path
        self._run_decisions_path: Optional[Path] = None
        self._run_stream_file = None
        if self._run_context:
            self._run_decisions_path = self._run_context.decisions_path / f"decisions_{tool}_{self.session_id}.jsonl"

        # Re-entrant lock guarding the per-decision append + multi-stream
        # write critical section in :meth:`log_decision`. Single-threaded
        # callers (the overwhelming majority) acquire an uncontended lock
        # per decision — byte-identical output, negligible overhead. Under
        # the opt-in concurrent rewrite-tier fanout
        # (``COURSEFORGE_REWRITE_CONCURRENCY > 1``) multiple worker threads
        # share ONE capture instance; this lock serializes the
        # ``self.decisions.append`` + ``_write_to_streams`` so concurrent
        # JSONL writes never interleave mid-line and no event is lost. The
        # POSIX ``flock`` in ``_write_to_streams`` is inter-PROCESS only (and
        # a silent no-op on WSL2 DrvFS), so it does not protect threads that
        # share a single file handle — this in-process lock does. RLock so a
        # nested ``log_decision`` (e.g. ``log_non_decision`` -> ``log_decision``)
        # on the same thread does not self-deadlock.
        self._log_lock = threading.RLock()

        # Initialize decision log
        self.decisions: List[Dict[str, Any]] = []
        self.prompts_responses: List[Dict[str, Any]] = []
        self.web_searches: List[Dict[str, Any]] = []
        self.files_created: List[str] = []
        self.sources_used: Dict[str, Any] = {}

        # Streaming file handles (dual-write, triple-write in hardened mode)
        self._stream_file = None
        self._legacy_stream_file = None
        self._stream_path = None
        self._legacy_stream_path = None
        if self.streaming_mode:
            self._stream_path = self.output_dir / f"decisions_{self.session_id}.jsonl"
            self._legacy_stream_path = self.legacy_output_dir / f"decisions_{self.session_id}.jsonl"
            try:
                self._stream_file = open(self._stream_path, 'a', encoding='utf-8')
            except OSError as e:
                logger.warning("Failed to open stream file %s: %s", self._stream_path, e)
                self.streaming_mode = False
                self._stream_file = None
            # Also open legacy stream file for dual-write
            try:
                self._legacy_stream_file = open(self._legacy_stream_path, 'a', encoding='utf-8')
            except OSError as e:
                logger.warning("Failed to open legacy stream file %s: %s", self._legacy_stream_path, e)
                self._legacy_stream_file = None
            # Phase 0: Also open run-specific stream file
            if self._run_decisions_path:
                try:
                    self._run_stream_file = open(self._run_decisions_path, 'a', encoding='utf-8')
                except OSError as e:
                    logger.warning("Failed to open run stream file %s: %s", self._run_decisions_path, e)
                    self._run_stream_file = None

        # Builder-2 decision-capture buffering (ED4ALL_CAPTURE_BUFFER). When
        # enabled, serialized JSONL rows accumulate in ``self._buffer`` under
        # ``self._log_lock`` and are batch-written+fsynced every
        # ``self._buffer_rows`` rows (and on flush/close/atexit) instead of one
        # fsync per decision per stream. Default OFF → byte-identical per-call
        # write path (see ``_write_to_streams``).
        self._buffer_enabled = _env_truthy("ED4ALL_CAPTURE_BUFFER")
        self._buffer_rows = _resolve_capture_buffer_rows()
        self._buffer: List[str] = []
        self._atexit_flush_fn = None
        if self._buffer_enabled and self.streaming_mode:
            # Register a weakref-backed atexit drain so a clean process exit
            # never loses buffered telemetry. The closure holds only a weakref
            # (no strong ref to self), so it never keeps the capture alive; a
            # GC'd or already-closed instance flushes as a no-op. Unregistered
            # in close() to keep the atexit registry bounded.
            self_ref = weakref.ref(self)

            def _atexit_flush(_ref=self_ref):
                inst = _ref()
                if inst is not None:
                    try:
                        inst.flush()
                    except Exception:
                        pass

            self._atexit_flush_fn = _atexit_flush
            atexit.register(self._atexit_flush_fn)

    def _build_storage(
        self, course_code: str, tool: str, phase: Optional[str]
    ) -> LibV2Storage:
        """Construct the LibV2 storage handle for decision capture.

        ROOT FIX (supersedes the W0.5 ``ED4ALL_COURSE_IDENTITY_DEDUP`` dedup
        layer): construct with ``auto_create=False`` so logging a decision
        NEVER creates an empty ``LibV2/courses/<slug>/`` skeleton (the recurring
        split-brain "empty twin"). Decision capture writes only to
        ``catalog/<course_id>/training/…``; :meth:`LibV2Storage.get_training_capture_path`
        (called immediately after this in ``__init__``) mkdir's its own phase
        dir under ``self.training_path`` — which is set unconditionally at
        ``LibV2Storage.__init__`` regardless of ``auto_create`` — so the CAPTURE
        write location + behavior are byte-identical to the legacy path; the
        only change is the absence of the collateral course skeleton
        (``imscc_chunks/`` / ``sources/`` / ``concept_graph/``).

        Because a decision capture no longer creates a twin, the
        ``ED4ALL_COURSE_IDENTITY_DEDUP`` branch that used to clean up its OWN
        collateral is obsolete here — the flag is now a no-op from
        ``decision_capture``. The ``lib/course_identity.py`` resolver + cleanup
        helpers are retained for any future caller that needs to consolidate
        PRE-EXISTING twins from other sources.
        """
        return LibV2Storage(course_code, auto_create=False)

    def _infer_operation(self, decision_type: str) -> str:
        """Infer operation from decision type for ML labeling."""
        return OPERATION_MAP.get(decision_type, f"decide_{decision_type}")

    # Wave4-I5: boilerplate-rationale detector thresholds.
    # Any decision_type with >= _BOILERPLATE_MIN_COUNT entries is scanned.
    # If > _BOILERPLATE_RATE_THRESHOLD of those entries share the same
    # first-_BOILERPLATE_PREFIX_LEN-char prefix, a structured WARNING is
    # emitted (not a decision_event — that would recurse). Finding 6 of
    # plans/dispatch-7-execution-inspection-2026-05.md: 62/63 source_selection
    # decisions in a real full-course run shared a byte-identical 165-char string.
    _BOILERPLATE_MIN_COUNT: int = 5
    _BOILERPLATE_RATE_THRESHOLD: float = 0.60
    _BOILERPLATE_PREFIX_LEN: int = 80

    def _scan_boilerplate_rationales(self) -> None:
        """Scan in-memory decisions for duplicate boilerplate rationales.

        Groups by decision_type; for each group with >= _BOILERPLATE_MIN_COUNT
        entries, computes the modal 80-char prefix and its share. Emits a
        structured WARNING when share > _BOILERPLATE_RATE_THRESHOLD.

        Runs once at close() time against the already-accumulated self.decisions
        list so there is zero per-decision overhead at call-site hot paths.
        """
        from collections import Counter

        # Group rationale prefixes by decision_type
        by_type: Dict[str, List[str]] = {}
        for rec in self.decisions:
            dt = rec.get("decision_type", "")
            rationale = rec.get("rationale") or ""
            by_type.setdefault(dt, []).append(rationale[: self._BOILERPLATE_PREFIX_LEN])

        for dt, prefixes in by_type.items():
            n = len(prefixes)
            if n < self._BOILERPLATE_MIN_COUNT:
                continue
            most_common_prefix, most_common_count = Counter(prefixes).most_common(1)[0]
            share = most_common_count / n
            if share > self._BOILERPLATE_RATE_THRESHOLD:
                logger.warning(
                    "DUPLICATE_BOILERPLATE_RATIONALE: phase=%s decision_type=%s "
                    "count=%d shared_prefix='%s...'",
                    self.phase,
                    dt,
                    n,
                    most_common_prefix[:60],
                )

    def close(self):
        """Explicitly close all stream file handles."""
        self._scan_boilerplate_rationales()
        # Builder-2: drain any buffered rows to disk BEFORE the handles close
        # so a clean shutdown never loses buffered telemetry.
        try:
            self.flush()
        except Exception:  # pragma: no cover - defensive; never block close()
            pass
        # Buffer is drained; drop the atexit hook so the registry stays bounded
        # across many phase-scoped captures.
        atexit_fn = getattr(self, "_atexit_flush_fn", None)
        if atexit_fn is not None:
            try:
                atexit.unregister(atexit_fn)
            except Exception:  # pragma: no cover
                pass
            self._atexit_flush_fn = None
        for attr in ('_stream_file', '_legacy_stream_file', '_run_stream_file'):
            fh = getattr(self, attr, None)
            if fh and not fh.closed:
                try:
                    fh.flush()
                    fh.close()
                except OSError:
                    pass
            setattr(self, attr, None)

    def _write_with_facade(self, line: str) -> bool:
        """
        Write decision to all locations using WriteFacade for atomicity.

        Phase 0.5: Centralized write discipline with transaction semantics.
        Uses append-only writes to avoid read-modify-write race conditions.

        Args:
            line: JSON line to write

        Returns:
            True if all writes succeeded
        """
        if not WRITE_FACADE_AVAILABLE or not is_write_facade_enforced():
            return False  # Fall back to legacy write

        # Collect all paths to write to
        paths = []
        if self._stream_path:
            paths.append(self._stream_path)
        if self._legacy_stream_path:
            paths.append(self._legacy_stream_path)
        if self._run_decisions_path:
            paths.append(self._run_decisions_path)

        if not paths:
            return False

        # Create WriteFacade for these paths
        allowed_dirs = [p.parent for p in paths]
        facade = WriteFacade(
            allowed_paths=allowed_dirs,
            enforce_allowed_paths=True,
        )

        # Use transaction for all-or-nothing semantics
        try:
            facade.begin_transaction()

            success = True
            for path in paths:
                # Append-only write: avoids read-modify-write race condition
                result = facade.append(path, line)
                if not result.success:
                    success = False
                    break

            if success:
                facade.commit_transaction()
                return True
            else:
                facade.rollback_transaction()
                return False

        except Exception as e:
            logger.warning("WriteFacade transaction failed: %s", e)
            try:
                facade.rollback_transaction()
            except Exception as rollback_err:
                logger.warning("WriteFacade rollback also failed: %s", rollback_err)
            return False

    def set_module_context(
        self,
        week: int,
        module: int,
        artifact_hash: Optional[str] = None
    ):
        """
        Set the current module context for ID generation.

        Args:
            week: Week number (1-16)
            module: Module number within week (1-3)
            artifact_hash: Optional hash of produced content
        """
        self.module_id = f"{self.course_id}_W{week:02d}_M{module:02d}"
        self.artifact_id = artifact_hash or hashlib.sha256(
            f"{self.module_id}_{self.session_id}".encode()
        ).hexdigest()[:12]

    def _build_record(
        self,
        decision_type: str,
        decision: str,
        rationale: str,
        operation: Optional[str],
        alternatives_considered: Optional[List[str]],
        context: Optional[str],
        confidence: Optional[float],
        ml_features: Optional[MLFeatures],
        inputs_ref: Optional[List[InputRef]],
        prompt_ref: Optional[str],
        outcome: Optional[OutcomeSignals],
        task_id: Optional[str],
        is_default: bool,
        outputs: Optional[List[OutputArtifact]],
        **kwargs
    ) -> Dict[str, Any]:
        """Build a decision record dict from parameters."""
        # A capture created outside RunManager still has a stable fallback
        # ``run_id``.  Scope the sequence manager to it as well; passing None
        # made every standalone/concurrent synthesis event receive seq=0.
        seq, event_id = get_sequence_for_context(self.run_id)

        rationale_length = len(rationale) if rationale else 0
        quality_level = self._assess_quality(rationale_length, inputs_ref, alternatives_considered, decision_type)
        effective_task_id = task_id or self.task_id

        record = {
            "event_id": event_id,
            "seq": seq,
            "run_id": self.run_id,
            "course_id": self.course_id,
            "module_id": self.module_id,
            "artifact_id": self.artifact_id,
            "task_id": effective_task_id,
            "tool": self.tool,
            "operation": operation or self._infer_operation(decision_type),
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase,
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "alternatives_considered": alternatives_considered or [],
            "context": context,
            "confidence": confidence,
            "is_default": is_default,
            "ml_features": _coerce_record_field(ml_features) or {},
            "inputs_ref": _coerce_record_field_list(inputs_ref),
            "prompt_ref": prompt_ref,
            "outputs": _coerce_record_field_list(outputs),
            "outcome": _coerce_record_field(outcome),
            "metadata": {
                "rationale_length": rationale_length,
                "quality_level": quality_level,
                "hardening_version": "1.0.0" if HARDENING_AVAILABLE else None,
                **kwargs
            }
        }
        fresh_start_id = getattr(self, "fresh_start_id", None)
        if fresh_start_id is not None:
            marker_digest = getattr(self, "fresh_start_marker_digest", None)
            record["metadata"]["fresh_start_id"] = fresh_start_id
            record["metadata"]["fresh_start_marker_digest"] = marker_digest
            try:
                context_value = json.loads(context) if context else {}
            except (TypeError, ValueError):
                context_value = {"detail": context}
            if not isinstance(context_value, dict):
                context_value = {"detail": context_value}
            context_value["fresh_start_id"] = fresh_start_id
            context_value["fresh_start_marker_digest"] = marker_digest
            record["context"] = json.dumps(
                context_value, sort_keys=True, separators=(",", ":"),
            )
        holdout_identity = getattr(self, "synthesis_holdout_identity", None)
        if holdout_identity is not None:
            identity = dict(holdout_identity)
            record["metadata"]["synthesis_holdout_identity"] = identity
            try:
                context_value = json.loads(record["context"]) if record["context"] else {}
            except (TypeError, ValueError):
                context_value = {"detail": record["context"]}
            if not isinstance(context_value, dict):
                context_value = {"detail": context_value}
            context_value["synthesis_holdout_identity"] = identity
            record["context"] = json.dumps(
                context_value, sort_keys=True, separators=(",", ":"),
            )
        run_contract = getattr(self, "synthesis_run_contract_sha256", None)
        if run_contract is not None:
            record["metadata"]["synthesis_run_contract_sha256"] = run_contract
            components = getattr(
                self, "synthesis_run_contract_components", None,
            )
            if components is not None:
                record["metadata"]["synthesis_run_contract_components"] = dict(
                    components
                )
            component_sha = getattr(
                self, "synthesis_contract_components_sha256", None,
            )
            if component_sha is not None:
                record["metadata"][
                    "synthesis_contract_components_sha256"
                ] = component_sha
            try:
                context_value = (
                    json.loads(record["context"]) if record["context"] else {}
                )
            except (TypeError, ValueError):
                context_value = {"detail": record["context"]}
            if not isinstance(context_value, dict):
                context_value = {"detail": context_value}
            context_value["synthesis_run_contract_sha256"] = run_contract
            record["context"] = json.dumps(
                context_value, sort_keys=True, separators=(",", ":"),
            )

        # Quality gating for training corpus filtering
        from .quality import check_quality_acceptable
        quality_ok, quality_reason = check_quality_acceptable(
            quality_level, minimum_level="proficient"
        )
        record["metadata"]["quality_gate_passed"] = quality_ok
        if not quality_ok:
            record["metadata"]["quality_gate_reason"] = quality_reason
            logger.warning(
                "Quality gate: decision '%s' rated '%s' (below proficient) "
                "— flagged for exclusion from training corpus",
                decision_type, quality_level,
            )

        return record

    def _validate_record(self, record: Dict[str, Any]) -> None:
        """Validate a decision record (REC-CTR-04 Worker G).

        Behavior matrix:

        * ``VALIDATE_DECISIONS`` unset/false -> no-op. Preserves backward
          compat for callers that explicitly opted out of validation entirely.
        * ``VALIDATE_DECISIONS`` truthy + ``DECISION_VALIDATION_STRICT`` unset
          -> warn-only. Validation issues are appended to
          ``record["metadata"]["validation_issues"]`` and the record IS still
          written by the caller. This is the historical default.
        * ``VALIDATE_DECISIONS`` truthy + ``DECISION_VALIDATION_STRICT=true``
          -> fail-closed. Validation failures raise ``ValueError`` and the
          record is NOT written (the caller must handle the exception).

        Opt-in strict mode is the reconciliation target for REC-CTR-04. The
        env-var gate preserves backward compat: callers relying on loose
        behavior do not break on this PR landing.
        """
        if not VALIDATE_DECISIONS:
            return

        strict = os.getenv("DECISION_VALIDATION_STRICT", "").lower() == "true"

        try:
            from .validation import validate_decision
            is_valid, issues = validate_decision(record, self.tool)
            if not is_valid:
                if strict:
                    raise ValueError(
                        f"Decision validation failed (strict mode): {issues}"
                    )
                # Wave 29 Defect 4: the non-strict validation-issues
                # path previously hit ``logger.warning`` on EVERY
                # record, flooding stderr with hundreds of
                # ``Decision validation issues: [...]`` lines on a
                # normal run (≥ 90% of stderr in the first 30
                # seconds of the SIM_RUN_01 reproduction). Real
                # errors got buried. The issues are still attached
                # to the record's metadata so they're recoverable at
                # save time (see ``_validation_issue_count`` +
                # ``save``) — we just stop spamming stderr per-call.
                # Strict mode still raises, so fail-closed callers
                # are unaffected.
                logger.debug("Decision validation issues: %s", issues)
                record["metadata"]["validation_issues"] = issues
                self._validation_issue_count = (
                    getattr(self, "_validation_issue_count", 0) + 1
                )
        except ImportError:
            pass  # Validation module not available
        except ValueError:
            raise  # Re-raise strict-mode failures
        except Exception as e:
            logger.debug("Decision validation error: %s", e)

    def _emit_to_stream_handles(self, data: str) -> None:
        """flock-guarded ``write + flush + fsync`` of ``data`` to every open
        stream handle.

        ``data`` is one serialized JSONL row (per-call path) OR a batch of
        already-newline-terminated rows concatenated (buffered flush path);
        both are byte-equivalent on disk since each row already ends in
        ``\\n``.

        Worker W3: acquire LOCK_EX around each write so concurrent writers
        sharing a JSONL path (multi-worker fanout, pinned RUN_ID re-runs,
        module-scoped captures sharing _run_decisions_path) cannot interleave
        records mid-line. OSError fallback covers WSL2 DrvFS / NFS /
        unsupported FS where flock silently no-ops or crashes.
        """
        for fh, label in [
            (self._stream_file, "decision stream"),
            (self._legacy_stream_file, "legacy stream"),
            (self._run_stream_file, "run stream"),
        ]:
            if fh:
                try:
                    if _fcntl is not None:
                        try:
                            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                            try:
                                fh.write(data)
                                fh.flush()
                                os.fsync(fh.fileno())
                            finally:
                                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
                        except OSError as lock_exc:
                            # WSL2 DrvFS / NFS / unsupported FS: degrade to
                            # unlocked write + warn so the audit trail still
                            # captures the event (concurrent writers may
                            # interleave on this filesystem).
                            logger.warning(
                                "decision-capture flock unavailable on %s: %s. "
                                "Falling back to unlocked write — concurrent "
                                "writers may interleave.",
                                getattr(fh, "name", label), lock_exc,
                            )
                            fh.write(data)
                            fh.flush()
                            os.fsync(fh.fileno())
                    else:
                        # Non-POSIX (Windows): pre-W3 unlocked behavior.
                        fh.write(data)
                        fh.flush()
                        os.fsync(fh.fileno())
                except OSError as e:
                    logger.warning("%s write failed: %s", label, e)

    def _flush_buffer_locked(self) -> None:
        """Batch-write + fsync all buffered rows. Caller MUST hold _log_lock.

        Concatenating buffered rows (each already ``\\n``-terminated) yields
        bytes identical to writing them one at a time, so the on-disk JSONL is
        byte-identical to the unbuffered path — only the fsync COUNT drops.
        """
        buf = getattr(self, "_buffer", None)
        if not buf:
            return
        blob = "".join(buf)
        buf.clear()
        self._emit_to_stream_handles(blob)

    def flush(self) -> None:
        """Drain any buffered decision rows to disk.

        No-op when ``ED4ALL_CAPTURE_BUFFER`` is off (nothing is ever buffered)
        or the buffer is empty. Safe to call from a phase-end seam, from
        close(), and from the atexit hook; acquires ``_log_lock`` so it is
        thread-safe against concurrent ``log_decision`` calls.
        """
        lock = getattr(self, "_log_lock", None)
        if lock is None:  # partial __init__ (e.g. __del__ after a failed ctor)
            return
        with lock:
            self._flush_buffer_locked()

    def _write_to_streams(self, record: Dict[str, Any]) -> None:
        """Write a decision record to all configured stream locations."""
        if not self.streaming_mode:
            return

        line = json.dumps(record) + '\n'

        # Builder-2: buffered path. Accumulate the serialized row and batch the
        # write+fsync every ``self._buffer_rows`` rows. ``_write_to_streams`` is
        # already called under ``self._log_lock`` from ``log_decision``; guard
        # the buffer with the same re-entrant lock so a direct call (e.g.
        # tests) or a crossblock worker thread stays thread-safe. Facade is
        # bypassed on this opt-in path (the batched legacy triple-write owns
        # write discipline here).
        if self._buffer_enabled:
            with self._log_lock:
                self._buffer.append(line)
                if len(self._buffer) >= self._buffer_rows:
                    self._flush_buffer_locked()
            return

        # Phase 0.5: Try WriteFacade first for atomic writes
        if self._write_with_facade(line):
            return

        # Fall back to legacy triple-write (per-call flock + fsync).
        self._emit_to_stream_handles(line)

    def log_decision(
        self,
        decision_type: str,
        decision: str,
        rationale: str,
        operation: Optional[str] = None,
        alternatives_considered: Optional[List[str]] = None,
        context: Optional[str] = None,
        confidence: Optional[float] = None,
        ml_features: Optional[MLFeatures] = None,
        inputs_ref: Optional[List[InputRef]] = None,
        prompt_ref: Optional[str] = None,
        outcome: Optional[OutcomeSignals] = None,
        task_id: Optional[str] = None,
        is_default: bool = False,
        outputs: Optional[List[OutputArtifact]] = None,
        **kwargs
    ):
        """Log a decision point with ML-trainable fields."""
        record = self._build_record(
            decision_type=decision_type,
            decision=decision,
            rationale=rationale,
            operation=operation,
            alternatives_considered=alternatives_considered,
            context=context,
            confidence=confidence,
            ml_features=ml_features,
            inputs_ref=inputs_ref,
            prompt_ref=prompt_ref,
            outcome=outcome,
            task_id=task_id,
            is_default=is_default,
            outputs=outputs,
            **kwargs
        )

        self._validate_record(record)
        # Serialize the append + multi-stream write so concurrent rewrite-tier
        # worker threads sharing this capture cannot interleave JSONL lines or
        # race the in-memory list. Uncontended (single-threaded) callers pay
        # only an atomic lock acquire/release — output stays byte-identical.
        with self._log_lock:
            self.decisions.append(record)
            self._write_to_streams(record)

    def _assess_quality(
        self,
        rationale_length: int,
        inputs_ref: Optional[List[InputRef]],
        alternatives: Optional[List[str]],
        decision_type: str = ""
    ) -> str:
        """Assess decision quality for training data filtering."""
        # Use centralized quality assessment from quality module
        rationale = "x" * rationale_length  # Dummy string with correct length
        return assess_decision_quality(rationale, inputs_ref, alternatives, decision_type)

    def log_non_decision(
        self,
        decision_type: str,
        default_value: str,
        rationale: str = "Default value used - no explicit decision required",
        context: Optional[str] = None,
        **kwargs
    ):
        """
        Log when a default is used instead of an explicit decision.

        Phase 0 Hardening: Captures "non-decisions" for training and reproducibility.

        Args:
            decision_type: Type of decision that was not explicitly made
            default_value: The default value that was applied
            rationale: Why the default was used
            context: Additional context
        """
        self.log_decision(
            decision_type=decision_type,
            decision=f"Used default: {default_value}",
            rationale=rationale,
            is_default=True,
            context=context,
            **kwargs
        )

    def log_outcome(
        self,
        artifact_id: str,
        accepted: bool = True,
        revision_count: int = 0,
        edit_distance: str = "none",
        quality_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log outcome signals for a produced artifact."""
        self.log_decision(
            decision_type="outcome_signal",
            decision=f"Artifact {artifact_id}: {'accepted' if accepted else 'rejected'}",
            rationale=f"Revisions: {revision_count}, Edit distance: {edit_distance}",
            operation="record_outcome",
            outcome=OutcomeSignals(
                accepted=accepted,
                revision_count=revision_count,
                edit_distance=edit_distance,
                quality_metrics=quality_metrics or {}
            )
        )

    def log_prompt_response(
        self,
        prompt: str,
        response: str,
        model: str = "claude",
        tokens_used: Optional[int] = None,
        purpose: str = ""
    ):
        """Log a Claude API prompt/response pair for SFT training."""
        self.prompts_responses.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "purpose": purpose,
            "prompt_summary": prompt[:500] if len(prompt) > 500 else prompt,
            "response_summary": response[:500] if len(response) > 500 else response,
            "full_prompt": prompt,
            "full_response": response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_used": tokens_used
        })

    def log_web_search(self, query: str, results_used: List[str], purpose: str = ""):
        """Log a web search performed."""
        self.web_searches.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results_used": results_used,
            "purpose": purpose
        })

    def log_file_created(self, filepath: str, description: str = ""):
        """Log a file that was created."""
        self.files_created.append({
            "filepath": filepath,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })

    def log_source_usage(
        self,
        existing_imscc: bool = False,
        imscc_files_reviewed: Optional[List[str]] = None,
        textbooks_referenced: Optional[List[str]] = None,
        external_urls: Optional[List[str]] = None
    ):
        """Log which sources were used."""
        self.sources_used = {
            "existing_imscc": existing_imscc,
            "imscc_files_reviewed": imscc_files_reviewed or [],
            "textbooks_referenced": textbooks_referenced or [],
            "external_urls": external_urls or []
        }

    def save(self, filename: Optional[str] = None) -> Path:
        """Save the decision capture to JSON file (triple-write in hardened mode)."""
        self.close()

        if filename is None:
            filename = f"decisions_{self.session_id}.json"

        output_path = self.output_dir / filename
        legacy_output_path = self.legacy_output_dir / filename

        # Wave 29 Defect 4: emit a single INFO-level summary line at
        # capture close instead of the hundreds of WARNING lines per
        # non-strict validation issue. Detail is still in the JSONL
        # captures under ``metadata.validation_issues``; pass ``-v``
        # to surface the per-record DEBUG lines.
        issue_count = getattr(self, "_validation_issue_count", 0)
        logger.info(
            "Captured %d decisions (%d with validation issues — "
            "run with -v for detail) [%s/%s]",
            len(self.decisions),
            issue_count,
            self.tool,
            self.phase,
        )

        data = {
            "course_code": self.course_code,
            "phase": self.phase,
            "tool": self.tool,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "decisions": self.decisions,
            "prompts_responses": self.prompts_responses,
            "web_searches": self.web_searches,
            "files_created": self.files_created,
            "sources_used": self.sources_used,
            "summary": {
                "total_decisions": len(self.decisions),
                "total_prompts": len(self.prompts_responses),
                "total_searches": len(self.web_searches),
                "total_files": len(self.files_created),
                "total_validation_issues": issue_count,
            }
        }

        # Atomic write to primary location
        temp_path = output_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as e:
                logger.warning("fsync failed on primary save: %s", e)
        os.rename(temp_path, output_path)

        # Also save to legacy location
        try:
            legacy_temp_path = legacy_output_path.with_suffix('.tmp')
            with open(legacy_temp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(legacy_temp_path, legacy_output_path)
        except OSError as e:
            logger.warning("Failed to save to legacy location: %s", e)

        return output_path

    def validate(self) -> Dict[str, Any]:
        """Validate that this capture has sufficient data."""
        result = {
            "valid": False,
            "course_code": self.course_code,
            "phase": self.phase,
            "decision_count": len(self.decisions),
            "issues": []
        }

        min_required = MIN_DECISIONS_PER_PHASE.get(self.phase, 1)
        if len(self.decisions) < min_required:
            result["issues"].append(
                f"Insufficient decisions: {len(self.decisions)} < {min_required} required"
            )

        decision_types = {d["decision_type"] for d in self.decisions}
        if self.phase == "content-generator" and "content_structure" not in decision_types:
            result["issues"].append("No content_structure decision logged")
        if self.phase == "input-research" and "source_selection" not in decision_types:
            result["issues"].append("No source_selection decision logged")

        result["valid"] = len(result["issues"]) == 0
        result["decision_types"] = list(decision_types)
        return result

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.save()
        except Exception as e:
            logger.error("Error saving decision capture: %s", e)
            if exc_type is None:
                raise
        return False


class SemantiKDecisionCapture(DecisionCapture):
    """Specialized decision capture for SemantiK PDF->HTML conversions.

    Labels captures with the ``semantik`` tool / ``semantik_conversion``
    phase.
    """

    def __init__(self, course_code: str, pdf_name: str):
        super().__init__(course_code, "semantik_conversion", "semantik")
        self.pdf_name = pdf_name
        self.conversion_details: Dict[str, Any] = {}

    def log_conversion_start(self, pdf_path: str, options: Dict[str, Any]):
        """Log the start of a SemantiK conversion."""
        self.conversion_details = {
            "pdf_path": pdf_path,
            "pdf_name": self.pdf_name,
            "started": datetime.now().isoformat(),
            "options": options
        }

    def log_conversion_complete(
        self,
        output_path: str,
        pages_processed: int,
        wcag_compliant: bool,
        processing_time_seconds: float
    ):
        """Log completion of a SemantiK conversion."""
        self.conversion_details.update({
            "completed": datetime.now().isoformat(),
            "output_path": output_path,
            "pages_processed": pages_processed,
            "wcag_compliant": wcag_compliant,
            "processing_time_seconds": processing_time_seconds
        })

    def log_structure_decision(
        self,
        page_range: str,
        detected_structure: str,
        applied_headings: List[str]
    ):
        """Log document structure detection decisions."""
        self.log_decision(
            decision_type="structure_detection",
            decision=f"Applied structure to pages {page_range}",
            rationale=f"Detected {detected_structure}",
            context=f"Headings applied: {', '.join(applied_headings)}"
        )

    def log_alt_text_decision(
        self,
        image_id: str,
        generated_alt_text: str,
        method: str = "claude"
    ):
        """Log alt text generation decisions."""
        self.log_decision(
            decision_type="alt_text_generation",
            decision=f"Generated alt text for {image_id}",
            rationale=f"Method: {method}",
            context=generated_alt_text[:200]
        )

    def log_math_decision(
        self,
        expression_id: str,
        original_text: str,
        mathml_output: str
    ):
        """Log math conversion decisions."""
        self.log_decision(
            decision_type="math_conversion",
            decision=f"Converted math expression {expression_id}",
            rationale="LaTeX to MathML for accessibility",
            context=f"Original: {original_text[:100]}"
        )

    def save(self, filename: Optional[str] = None) -> Path:
        """Save with conversion-specific details using atomic write."""
        self.close()

        if filename is None:
            filename = f"semantik_conversion_{self.pdf_name}_{self.session_id}.json"

        output_path = self.output_dir / filename
        legacy_output_path = self.legacy_output_dir / filename

        data = {
            "course_code": self.course_code,
            "phase": self.phase,
            "tool": self.tool,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "conversion_details": self.conversion_details,
            "decisions": self.decisions,
            "prompts_responses": self.prompts_responses,
            "summary": {
                "total_decisions": len(self.decisions),
                "total_prompts": len(self.prompts_responses),
                "pdf_name": self.pdf_name
            }
        }

        # Atomic write to primary location
        temp_path = output_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as e:
                logger.warning("fsync failed on SemantiK-capture primary save: %s", e)
        os.rename(temp_path, output_path)

        # Also save to legacy location
        try:
            legacy_temp_path = legacy_output_path.with_suffix('.tmp')
            with open(legacy_temp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(legacy_temp_path, legacy_output_path)
        except OSError as e:
            logger.warning("Failed to save SemantiK capture to legacy location: %s", e)

        return output_path


def create_capture(course_code: str, phase: str, tool: str = "courseforge") -> DecisionCapture:
    """Factory function to create a decision capture instance."""
    return DecisionCapture(course_code, phase, tool)


def create_semantik_capture(course_code: str, pdf_name: str) -> SemantiKDecisionCapture:
    """Factory function to create a SemantiK conversion decision capture."""
    return SemantiKDecisionCapture(course_code, pdf_name)
