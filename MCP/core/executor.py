#!/usr/bin/env python3
"""
Orchestrator Task Executor

Executes workflow tasks by mapping agent types to MCP tools.

Pipeline Position:
    Workflow Tasks → [Executor] → MCP Tools → Results

Decision Capture:
    All execution decisions logged for orchestration training.

Phase 0 Hardening:
    - Error classification for intelligent retry decisions
    - Poison-pill detection to stop bad batches
    - Phase checkpointing for crash recovery
    - Validation gates for quality assurance
"""

import asyncio
import contextvars
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

# Add project path
_CORE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CORE_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import STATE_PATH, get_state_runs_dir  # noqa: E402
# Workflow-state write discipline: atomic temp+replace + bounded advisory
# flock around the read-modify-write cycle (2026-07-21 corruption incident:
# two racing writers interleaved plain open('w')+dump partial writes).
from lib.file_lock import file_lock  # noqa: E402
from lib.state_manager import atomic_write_json  # noqa: E402
from lib.generation.stop_control import (  # noqa: E402
    GracefulStopRequested,
    clear_stop,
    request_stop,
    stop_requested,
)

from .config import OrchestratorConfig  # noqa: E402
from .param_mapper import ParameterMappingError, TaskParameterMapper  # noqa: E402

# Phase 0 Hardening: Import hardening modules with graceful fallback.
#
# These modules live in ``MCP/hardening/``, NOT ``MCP/core/`` — a relative
# ``from .error_classifier import ...`` silently hits the ``except ImportError``
# arm, flips every ``HARDENING_*`` flag to False, and no-ops the whole Phase 0
# stack at runtime (tests importing ``MCP.hardening.*`` directly do not catch
# it). Keep the absolute ``..hardening.*`` form. ``except ImportError`` is
# retained for deployments that strip the hardening package; the debug log keeps
# a silent regression observable.
try:
    from ..hardening.error_classifier import (
        ErrorClass,
        ErrorClassifier,
        PoisonPillDetector,
        RetryPolicy,
    )
    HARDENING_ERROR_CLASSIFIER = True
except ImportError as _exc:
    HARDENING_ERROR_CLASSIFIER = False
    ErrorClass = None
    RetryPolicy = None  # type: ignore[assignment]
    logging.getLogger(__name__).debug(
        "Hardening import failed (error_classifier): %s", _exc
    )

try:
    from ..hardening.checkpoint import CheckpointManager, PhaseCheckpoint  # noqa: F401
    HARDENING_CHECKPOINTS = True
except ImportError as _exc:
    HARDENING_CHECKPOINTS = False
    logging.getLogger(__name__).debug(
        "Hardening import failed (checkpoint): %s", _exc
    )

try:
    from ..hardening.validation_gates import (  # noqa: F401
        GateConfig,
        GateIssue,
        GateResult,
        GateSeverity,
        ValidationGateManager,
    )
    HARDENING_VALIDATION_GATES = True
except ImportError as _exc:
    HARDENING_VALIDATION_GATES = False
    logging.getLogger(__name__).debug(
        "Hardening import failed (validation_gates): %s", _exc
    )

try:
    from ..hardening.gate_input_routing import GateInputRouter, default_router
    HARDENING_GATE_INPUT_ROUTING = True
except ImportError as _exc:
    HARDENING_GATE_INPUT_ROUTING = False
    GateInputRouter = None  # type: ignore
    default_router = None  # type: ignore
    logging.getLogger(__name__).debug(
        "Hardening import failed (gate_input_routing): %s", _exc
    )

try:
    from ..hardening.lockfile import LockfileManager  # noqa: F401
    HARDENING_LOCKFILE = True
except ImportError as _exc:
    HARDENING_LOCKFILE = False
    logging.getLogger(__name__).debug(
        "Hardening import failed (lockfile): %s", _exc
    )

# Aggregate flag — True only when every Phase 0 hardening submodule
# imported cleanly. Consumers / regression tests assert against this
# single value rather than the four leaf flags.
HARDENING_PHASE_0 = (
    HARDENING_ERROR_CLASSIFIER
    and HARDENING_CHECKPOINTS
    and HARDENING_VALIDATION_GATES
    and HARDENING_LOCKFILE
)

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT TYPE TO MCP TOOL MAPPING
# =============================================================================
# Maps agent types (from config/agents.yaml) to actual MCP tool names.
# All tools listed here MUST exist in the MCP tool registry.
# =============================================================================

AGENT_TOOL_MAPPING = {
    # -------------------------------------------------------------------------
    # COURSEFORGE AGENTS
    # -------------------------------------------------------------------------
    # course-outliner routes to plan_course_structure (real LO synthesis +
    # persistence), NOT create_course_project (which only makes subdirs and
    # emits {COURSE}_OBJ_N placeholders). plan_course_structure must stay
    # robust to a missing textbook structure — the course_generation workflow
    # reaches it with only an objectives JSON.
    "course-outliner": "plan_course_structure",
    "requirements-collector": "get_courseforge_status",
    "content-generator": "generate_course_content",
    "brightspace-packager": "package_imscc",
    # WCAG/OSCQR validation runs as validation GATES (``lib.validators.wcag`` /
    # ``lib.validators.oscqr``), not as a tool — so these evaluator agents route
    # to the benign status tool.
    "oscqr-course-evaluator": "get_courseforge_status",
    "quality-assurance": "get_courseforge_status",

    # -------------------------------------------------------------------------
    # PIPELINE AGENTS (Textbook-to-Course)
    # -------------------------------------------------------------------------
    # textbook-ingestor routes to extract_textbook_structure (real
    # SemanticStructureExtractor dispatch), not create_course_project.
    "textbook-stager": "stage_semantik_outputs",
    "textbook-ingestor": "extract_textbook_structure",
    "source-router": "build_source_module_map",
    # Backs the ``concept_extraction`` phase; ``_run_concept_extraction``
    # (registered in ``MCP/tools/pipeline_tools.py::_build_tool_registry``)
    # produces the concept graph + manifest.
    "pedagogy-graph-builder": "run_concept_extraction",
    # Backs the ``chunking`` phase. Utility-style agent (no LLM dispatch);
    # ``run_dart_chunking`` emits ``LibV2/courses/<slug>/dart_chunks/
    # chunks.jsonl`` + ``manifest.json`` via ``Trainforge.chunker.chunk_content``.
    # The legacy ``dart-chunker`` key stays as a read-compat dispatch alias so
    # resume states / configs written before the rename still route.
    "semantik-chunker": "run_dart_chunking",
    "dart-chunker": "run_dart_chunking",  # legacy alias (read-compat)

    # -------------------------------------------------------------------------
    # CONVERSION / REMEDIATION AGENTS
    # -------------------------------------------------------------------------
    # ``semantik-converter`` backs the ``semantik_conversion`` phase;
    # ``extract_and_convert_pdf`` routes it to the SemantiK v2 cascade seam.
    # The legacy ``dart-*`` keys stay as read-compat dispatch aliases so resume
    # states / configs written before the rename still route.
    "semantik-automation-coordinator": "extract_and_convert_pdf",
    "semantik-converter": "extract_and_convert_pdf",
    "dart-automation-coordinator": "extract_and_convert_pdf",  # legacy alias
    "dart-converter": "extract_and_convert_pdf",  # legacy alias (read-compat)
    "imscc-intake-parser": "intake_imscc_package",
    "content-analyzer": "analyze_imscc_content",
    "accessibility-remediation": "remediate_course_content",
    "content-quality-remediation": "remediate_course_content",
    "intelligent-design-mapper": "remediate_course_content",
    "remediation-validator": "get_courseforge_status",

    # -------------------------------------------------------------------------
    # TRAINFORGE AGENTS
    # -------------------------------------------------------------------------
    "assessment-extractor": "analyze_imscc_content",
    # rag-indexer must route to ``run_vector_indexing`` (real embeddings +
    # numpy exact-search index), which FAILS CLOSED when the embedding backend
    # is unavailable. Do NOT point it at ``analyze_imscc_content``: that is an
    # HTML/word-count scan, so the ``rag_training`` ``indexing`` phase would
    # report success without building any index.
    "rag-indexer": "run_vector_indexing",
    "assessment-generator": "generate_assessments",
    "assessment-validator": "validate_assessment",
    "training-synthesizer": "synthesize_training",

    # -------------------------------------------------------------------------
    # LIBV2 AGENTS
    # -------------------------------------------------------------------------
    "libv2-archivist": "archive_to_libv2",
}


# =============================================================================
# Phase-name-aware tool dispatch
# =============================================================================
# Maps workflow phase names to MCP tool names. The dispatcher checks
# ``_PHASE_TOOL_MAPPING.get(phase)`` BEFORE falling back to
# ``AGENT_TOOL_MAPPING.get(agent_type)``, so these phases reach their dedicated
# handlers regardless of which agent name is threaded through the task.
# Membership here is also what lets a validator-only phase (``agents: []``) run
# at all: ``workflow_runner._create_phase_tasks`` synthesizes its single virtual
# ``phase-handler`` task only for phases present in this map.
# =============================================================================

_PHASE_TOOL_MAPPING: Dict[str, str] = {
    "content_generation_outline": "run_content_generation_outline",
    "inter_tier_validation": "run_inter_tier_validation",
    "content_generation_rewrite": "run_content_generation_rewrite",
    "post_rewrite_validation": "run_post_rewrite_validation",
    # Both chunking phases share one content-agnostic chunker agent, but emit
    # differently: the IMSCC-side tool writes ``imscc_chunks/`` with
    # ``chunkset_kind="imscc"`` + ``source_imscc_sha256``, the DART-side tool
    # (reached via the AGENT_TOOL_MAPPING fallback) writes ``dart_chunks/`` with
    # ``chunkset_kind="dart"`` + ``source_dart_html_sha256``. The phase-name
    # override picks the right helper without forking the agent registry.
    "imscc_chunking": "run_imscc_chunking",
    # Pre-packaging assessment synthesis: emits QTI 1.2 / imsdt / assignment XML
    # into ``<export>/06_assessments/`` for the packaging phase to consume.
    "assessment_synthesis": "run_assessment_synthesis",
    # Post-conversion heading-level judge over the GLM-OCR lane's
    # ``{stem}.glmocr_layout.json`` sidecars, between ``semantik_conversion``
    # and ``staging``. Shells out per chapter, copies judged HTML/escalations
    # back over the conversion output (keeping ``.prejudge.bak`` / ``.bak``),
    # and FAIL-OPENS per chapter — a judge failure must never block a build.
    # SEMANTIK_HEADING_JUDGE is DEFAULT-ON (explicit falsey token opts out);
    # skip-with-pass when explicitly off or no sidecars exist (born-digital).
    "heading_judge": "run_heading_judge",
    # ``trainforge_train``. The training phase declares the
    # ``training-synthesizer`` agent, whose AGENT_TOOL_MAPPING entry is
    # ``synthesize_training`` — the instruction/preference PAIR-SYNTHESIS tool.
    # On the agent fallback the workflow therefore re-synthesized training pairs
    # and reported the phase complete without ever calling a trainer, so
    # ``ed4all run trainforge_train`` trained nothing. Only a phase-name
    # override can reach the real handler without repurposing the agent (which
    # the training_synthesis phase of textbook_to_course still needs).
    "training": "run_training",
    # Post-training evaluation: rolls the held-out matrix + the grounded-answer
    # arm into one verdict and merges both into the eval_report.json that
    # ``post_training_validation``'s eval_gating gate reads.
    "evaluation": "run_evaluation",
    # NB: ``post_training_validation`` is deliberately ABSENT. It is a
    # validator-only phase (``agents: []``) like ``inter_tier_validation`` /
    # ``post_rewrite_validation``, but those two are mapped because each has a
    # dedicated Python handler to run BEFORE its gates. post_training_validation
    # has no handler — its ``eval_gating`` / ``family_completeness`` gates run
    # through the gate manager at phase end, which needs no task. A mapping here
    # would synthesize a virtual phase-handler task pointing at nothing.
}


#: Registry tools that must NEVER route through the subagent fork, keyed by the
#: RESOLVED tool name rather than the agent name (see ``_invoke_tool``).
_DETERMINISTIC_TRAINING_TOOLS: frozenset = frozenset({
    "run_training",
    "run_evaluation",
})


# =============================================================================
# Agent classification for per-task subagent dispatch
# =============================================================================
#
# Every ``AGENT_TOOL_MAPPING`` entry resolves to a Python tool that
# ``TaskExecutor._invoke_tool`` calls in-process. This set additionally splits
# agents into:
#
#   * **subagent-dispatched** (listed here) — work that genuinely needs LLM
#     reasoning. When ``ED4ALL_AGENT_DISPATCH`` is truthy AND a dispatcher is
#     threaded into the executor, these route through
#     ``dispatcher.dispatch_task`` instead of the in-process tool, and a
#     subagent on the other end of the mailbox bridge does the work per that
#     agent's spec file.
#
#   * **Python-tool** (absent here) — deterministic work (extraction, TF-IDF
#     routing, staging, packaging, static validation, archival). These stay on
#     ``_invoke_tool`` regardless of the flag. Their LLM sub-calls (e.g. alt
#     text, block classification) still go through ``MailboxBrokeredBackend``;
#     that is orthogonal to this classification.
#
# Membership is an explicit list, not derived from the agent spec's prose
# style: an agent that reads like a reasoning agent but is backed by a Python
# tool (or vice-versa) must only flip classification under deliberate review.
AGENT_SUBAGENT_SET = frozenset({
    # Courseforge reasoning agents
    "course-outliner",         # LO synthesis from textbook structure
    "content-generator",       # weekly module page emission
    "oscqr-course-evaluator",  # OSCQR rubric evaluation (subjective)
    "quality-assurance",       # pattern prevention & validation narrative

    # Remediation reasoning agents (HTML enhancement)
    "content-analyzer",                # accessibility + quality gap detection
    "accessibility-remediation",       # alt-text, heading hierarchy fixes
    "content-quality-remediation",     # educational depth enhancement
    "intelligent-design-mapper",       # component selection + styling

    # Trainforge reasoning agents
    "assessment-extractor",            # narrative content-extraction summaries
    "assessment-generator",            # question + distractor generation
    "assessment-validator",            # alignment + rubric judgments
    "training-synthesizer",            # instruction + preference pair synthesis
})


# Agent → provider-env-var mapping driving the in-process short-circuits below.
# Kept as a map rather than inline ``os.environ.get(...)`` checks so
# ``workflow_runner`` can emit its provider banner without re-stating the
# literals, and new provider plumbing is one entry rather than a duplicated
# ``_force_inprocess_for_*`` triple.
AGENT_PROVIDER_ENV_MAP: Mapping[str, str] = {
    "content-generator": "COURSEFORGE_PROVIDER",
    "course-outliner": "COURSEPLANNER_PROVIDER",
    "assessment-generator": "TRAINFORGE_ASSESSMENT_PROVIDER",
}


# The COMPLETE map of subagent-classified agents (``AGENT_SUBAGENT_SET``) to the
# env var that, when set, short-circuits mailbox subagent dispatch and routes
# that agent's LLM work through the in-process OpenAI-compatible provider
# lattice. Single source of truth for "which env var blesses each LLM agent into
# the in-process lattice", consumed by
# ``workflow_runner._enforce_authoring_provider_route`` so headless/GUI runs FAIL
# FAST instead of hanging on an unserviced mailbox or silently degrading to a
# templated stub.
#
# ``AGENT_PROVIDER_ENV_MAP`` above is a strict subset (three agents);
# ``training-synthesizer``'s ``TRAINFORGE_SYNTHESIS_PROVIDER`` short-circuit is
# still inline in ``_invoke_tool``.
#
# The other subagent-classified agents (oscqr-course-evaluator,
# quality-assurance, content-analyzer, accessibility-remediation,
# content-quality-remediation, intelligent-design-mapper, assessment-extractor,
# assessment-validator) have NO provider short-circuit — they can only run via
# session subagent dispatch, and none appear in ``textbook_to_course``, so the
# four entries below fully cover the textbook authoring route. The guardrail
# treats agents absent from this map as session-only: dispatching one without a
# servicer still fails fast, but the fix is "run inside a session", not "set a
# provider env".
AGENT_AUTHORING_PROVIDER_ENV_MAP: Mapping[str, str] = {
    "content-generator": "COURSEFORGE_PROVIDER",
    "course-outliner": "COURSEPLANNER_PROVIDER",
    "assessment-generator": "TRAINFORGE_ASSESSMENT_PROVIDER",
    "training-synthesizer": "TRAINFORGE_SYNTHESIS_PROVIDER",
}


# Feature flag enabling the dispatch_task routing fork. Default off. Evaluated
# per-call so tests can toggle via ``monkeypatch.setenv``.
_AGENT_DISPATCH_ENV = "ED4ALL_AGENT_DISPATCH"


def _agent_dispatch_enabled() -> bool:
    """Return True iff ``ED4ALL_AGENT_DISPATCH`` is set to a truthy value.

    Read at call time (not import) so tests can toggle the flag per-run.
    Accepts ``1``, ``true``, ``yes``, ``on`` (case-insensitive). Anything
    else — including unset — is treated as off.
    """
    raw = os.environ.get(_AGENT_DISPATCH_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Graceful-stop timeout → pause conversion
# --------------------------------------------------------------------------- #
# When a wall-clock deadline expires, the executor does NOT immediately cancel
# the in-flight coroutine (``asyncio.wait_for`` would — that is exactly the hard
# kill the "checkpoint on command" grace period must prevent). Instead it grants
# a bounded GRACE window for the workers to reach their next unit boundary,
# checkpoint the in-flight unit, and return a PAUSED result. Only if the grace
# window ALSO expires (an unresponsive worker that never consults the stop
# channel) does it hard-cancel and fall through to the pre-existing TIMEOUT
# handling. The grace is a fraction of the deadline, capped — expressed as
# module CONSTANTS (never an env var: this is internal mechanics, NOT a new
# behavior flag) so tests can monkeypatch them to small numbers.
BATCH_TIMEOUT_GRACE_FRACTION = 0.10
BATCH_TIMEOUT_GRACE_CAP_SECONDS = 600.0


def _grace_seconds(timeout_seconds: float) -> float:
    """Bounded drain window after a wall-clock deadline (``min(cap, 10%·T)``).

    Read from the module constants at call time so a test can monkeypatch
    :data:`BATCH_TIMEOUT_GRACE_FRACTION` / :data:`BATCH_TIMEOUT_GRACE_CAP_SECONDS`
    and observe the change. Returns a float rather than an int floor so
    sub-second windows stay testable.
    """
    try:
        t = float(timeout_seconds)
    except (TypeError, ValueError):
        return 0.0
    return min(BATCH_TIMEOUT_GRACE_CAP_SECONDS, BATCH_TIMEOUT_GRACE_FRACTION * t)


# Task-scoped in-process stop channel. Distinct from the run-scoped
# filesystem sentinel in ``lib.generation.stop_control``: when a SINGLE task
# exceeds ``ED4ALL_TASK_TIMEOUT_MINUTES`` the executor sets this per-task Event
# and grants a grace window before a hard cancel — but it NEVER writes the
# run-scoped sentinel, because one slow task must not pause the whole run. A
# ``threading.Event`` (not ``asyncio.Event``) so a tool that offloads to
# ``asyncio.to_thread`` can consult it from its worker thread; the ContextVar is
# copied into the tool's task/thread context at dispatch. Stop-aware in-process
# tools MAY consult :func:`current_task_stop_event` to drain early; today the
# channel is grace-only (no in-process tool reads it yet), and after grace the
# existing TIMEOUT classification + transient-retry ladder stands unchanged
# (retry replays the fingerprinted resume sidecar, so no work is lost).
_TASK_STOP_EVENT: "contextvars.ContextVar[Optional[threading.Event]]" = (
    contextvars.ContextVar("ed4all_task_stop_event", default=None)
)


def current_task_stop_event() -> Optional[threading.Event]:
    """Return the in-process per-task stop Event for the current tool call.

    A stop-aware in-process tool MAY consult this to drain to its next unit
    boundary early once the executor signals a per-task timeout grace window.
    It is NOT the run-scoped filesystem sentinel (see the module note on
    :data:`_TASK_STOP_EVENT`) — consulting it can only shorten one slow task,
    never pause the run. Returns ``None`` outside a managed tool invocation.
    """
    return _TASK_STOP_EVENT.get()


# CUDA-OOM detection + free-VRAM probe live in the shared ``lib.llm.oom`` module
# so the validation-gate path (``MCP/hardening/validation_gates.py``) can
# recognise an OOM raised inside a validator WITHOUT a circular import (the
# executor imports validation_gates). The module-level ``_is_cuda_oom`` /
# ``_probe_free_vram_mib`` aliases are load-bearing: call sites and tests patch
# ``MCP.core.executor._probe_free_vram_mib``.
#
# Scope note for the executor's use of ``_is_cuda_oom``: here it governs
# LOGGING ONLY. The OOM branch in ``_execute_with_retries`` uses it to decide
# whether to emit the loud GPU-OOM diagnostic, then FALLS THROUGH to the
# unchanged ``ErrorClassifier`` path — a CUDA OOM message already matches the
# classifier's ``out of memory`` POISON_PATTERN, so it is classified
# POISON_PILL and stops the batch via the runaway-VRAM circuit breaker. Because
# it gates only the extra log line, an over-broad match here is harmless.
from lib.llm.oom import is_cuda_oom as _is_cuda_oom  # noqa: E402
from lib.llm.oom import probe_free_vram_mib as _probe_free_vram_mib  # noqa: E402


@dataclass
class ExecutionResult:
    """Result of executing a task."""
    task_id: str
    status: str  # "COMPLETE", "ERROR", "TIMEOUT", "POISON_PILL"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    # Phase 0 Hardening: Error classification
    error_class: Optional[str] = None  # "transient", "permanent", "poison_pill"
    retry_count: int = 0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "error_class": self.error_class,
            "retry_count": self.retry_count,
            "artifacts": self.artifacts,
        }


class ToolRegistryError(Exception):
    """Raised when tool registry validation fails."""
    pass


class TaskExecutor:
    """
    Executes workflow tasks by invoking MCP tools.

    Maps agent types to appropriate tools and handles:
    - Task dispatch and tracking
    - Result collection
    - Error handling and retries
    - Decision capture for training

    Usage:
        executor = TaskExecutor(tool_registry, capture=capture)
        executor.validate_tool_registry()  # Fail-fast check
        result = await executor.execute_task(workflow_id, task_id)
    """

    def __init__(
        self,
        tool_registry: Optional[Dict[str, Callable[..., Awaitable[str]]]] = None,
        capture: Optional["DecisionCapture"] = None,
        config: Optional[OrchestratorConfig] = None,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        validate_registry: bool = False,
        run_id: Optional[str] = None,
        run_path: Optional[Path] = None,
        poison_pill_threshold: int = 3,
        batch_timeout_minutes: Optional[int] = None,
        dispatcher: Optional[Any] = None,
    ):
        """
        Initialize the task executor.

        Args:
            tool_registry: Dict mapping tool names to async functions
            capture: Optional DecisionCapture for logging decisions
            config: Optional OrchestratorConfig (loaded from YAML if not provided)
            max_retries: Override for max retry attempts (uses config if not set)
            timeout_seconds: Override for task timeout (uses config if not set)
            validate_registry: If True, validate tool registry at startup (fail-fast)
            run_id: Unique run identifier for tracing. Auto-generated if not provided.
            run_path: Path to run directory for checkpoints (Phase 0 hardening)
            poison_pill_threshold: N same-pattern failures stops batch (Phase 0)
            batch_timeout_minutes: Timeout for entire batch (Phase 0)
            dispatcher: Optional dispatcher exposing a
                ``dispatch_task(*, task_name, agent_type, task_params,
                run_id, phase_context) -> dict`` coroutine. When present
                and ``ED4ALL_AGENT_DISPATCH`` is truthy and the task's
                agent_type is in ``AGENT_SUBAGENT_SET``, ``_invoke_tool``
                routes through the dispatcher instead of the in-process
                ``tool_registry`` entry. ``None`` keeps the in-process
                path for tests / direct instantiation.
        """
        self.tool_registry = tool_registry or {}
        self.capture = capture
        self.dispatcher = dispatcher
        # Graceful stop ("checkpoint on command"): the phase currently being
        # executed, published by ``execute_phase`` so ``_execute_parallel`` /
        # ``_run_and_record`` can per-task-checkpoint without a signature
        # change. ``None`` outside a phase (the ``execute_workflow`` path).
        self._active_phase_name: Optional[str] = None

        # Generate or use provided run_id for tracing
        self.run_id = run_id or os.environ.get(
            'RUN_ID',
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Load config if not provided
        try:
            self.config = config or OrchestratorConfig.load()
        except Exception as e:
            logger.warning(f"[{self.run_id}] Failed to load config, using defaults: {e}")
            self.config = OrchestratorConfig()

        # Use provided values or fall back to config
        self.max_retries = max_retries if max_retries is not None else self.config.retry_attempts
        # Per-task timeout. Like the batch timeout below, the config default
        # (60 min) is too tight for a slow local-7B phase that runs as a single
        # long task (the rewrite handler loops all blocks internally). Allow an
        # ``ED4ALL_TASK_TIMEOUT_MINUTES`` override (read at construction, so it
        # applies to resumes) when no explicit timeout_seconds is supplied.
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        else:
            _task_min = self.config.task_timeout_minutes
            _env_task_min = os.environ.get("ED4ALL_TASK_TIMEOUT_MINUTES")
            if _env_task_min:
                try:
                    _parsed_t = int(_env_task_min)
                    if _parsed_t > 0:
                        _task_min = _parsed_t
                except (TypeError, ValueError):
                    pass
            self.timeout_seconds = _task_min * 60
        # Batch (whole-phase) timeout default. This is the executor-wide
        # FALLBACK used by ``execute_phase`` when the phase config carries no
        # per-phase ``batch_timeout_minutes``. The per-phase YAML value IS now
        # plumbed through: ``WorkflowRunner.run_workflow`` reads
        # ``WorkflowPhase.batch_timeout_minutes`` (parsed from workflows.yaml)
        # and passes it to ``execute_phase`` as ``phase_batch_timeout_minutes``,
        # which wins over this fallback for that call. Resolution precedence in
        # ``execute_phase``: per-phase YAML ``batch_timeout_minutes`` (if set)
        # → ``ED4ALL_BATCH_TIMEOUT_MINUTES`` env → 30. So a phase that declares
        # ``batch_timeout_minutes: 240`` (a slow local-7B rewrite) gets 14400s
        # even when the env is unset. The env knob remains the way to widen a
        # phase that has NO YAML value (e.g. content_generation grinding
        # per-block CURIE-preservation retries). Read at construction time so it
        # applies to resumes too.
        _default_batch_min = 30
        _env_batch_min = os.environ.get("ED4ALL_BATCH_TIMEOUT_MINUTES")
        if _env_batch_min:
            try:
                _parsed = int(_env_batch_min)
                if _parsed > 0:
                    _default_batch_min = _parsed
            except (TypeError, ValueError):
                pass
        self.batch_timeout_seconds = (batch_timeout_minutes or _default_batch_min) * 60

        # Initialize parameter mapper
        self.param_mapper = TaskParameterMapper(strict=False)

        # Phase 0 Hardening: Initialize hardening components
        # Honor ED4ALL_STATE_RUNS_DIR override so unit tests can
        # redirect run state into tmp_path (see conftest.py
        # ``state_runs_isolated`` fixture).
        self.run_path = run_path or (get_state_runs_dir() / self.run_id)
        self._init_hardening(poison_pill_threshold)

        # Log initialization with run_id
        logger.info(f"[{self.run_id}] TaskExecutor initialized with {len(self.tool_registry)} tools")

        # Fail-fast validation if requested
        if validate_registry and self.tool_registry:
            self.validate_tool_registry()

    def _init_hardening(self, poison_pill_threshold: int) -> None:
        """Initialize Phase 0 hardening components."""
        # Error classifier for intelligent retry decisions
        self.error_classifier = None
        self.poison_detector = None
        self.retry_policy = None
        if HARDENING_ERROR_CLASSIFIER:
            self.error_classifier = ErrorClassifier()
            self.poison_detector = PoisonPillDetector(
                threshold=poison_pill_threshold,
                window_seconds=300
            )
            # RetryPolicy is what makes ``_execute_with_retries`` actually sleep
            # between attempts on transient errors; base_delay / max_delay /
            # exponential_base come from OrchestratorConfig's
            # ``retry_delay_seconds``.
            base_delay = float(
                getattr(self.config, "retry_delay_seconds", 5) or 5
            )
            self.retry_policy = RetryPolicy(
                max_retries=self.max_retries,
                base_delay_seconds=base_delay,
                max_delay_seconds=max(300.0, base_delay * 60),
                exponential_base=2.0,
            )
            logger.debug(
                f"[{self.run_id}] Error classifier + poison detector + "
                f"retry policy initialized (base_delay={base_delay}s)"
            )

        # Checkpoint manager for crash recovery
        self.checkpoint_manager = None
        if HARDENING_CHECKPOINTS and self.run_path:
            try:
                self.checkpoint_manager = CheckpointManager(self.run_path)
                logger.debug(f"[{self.run_id}] Checkpoint manager initialized")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to init checkpoint manager: {e}")

        # Validation gate manager
        self.gate_manager = None
        if HARDENING_VALIDATION_GATES:
            # H3 Worker S0.5: thread the executor's capture into the gate
            # manager so the direct-invocation `run_gate` path (test
            # harness / future MCP-exposed validate tools) also injects
            # `decision_capture` / `capture` keys for emitting validators.
            self.gate_manager = ValidationGateManager(capture=self.capture)
            logger.debug(f"[{self.run_id}] Validation gate manager initialized")

        # Per-gate input router: builds per-validator kwargs from the phase's
        # accumulated outputs + workflow params. Handing every gate one generic
        # ``{'artifacts': ..., 'results': ...}`` blob instead makes critical
        # gates return MISSING_INPUT and warning gates silently pass.
        self.gate_input_router = None
        if HARDENING_GATE_INPUT_ROUTING and default_router is not None:
            self.gate_input_router = default_router()
            logger.debug(f"[{self.run_id}] Gate input router initialized")

        # Lock manager for cross-process resource locking.
        self.lock_manager = None
        if HARDENING_LOCKFILE and self.run_path:
            try:
                self.lock_manager = LockfileManager(self.run_path)
                logger.debug(f"[{self.run_id}] Lock manager initialized")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to init lock manager: {e}")

    def validate_tool_registry(self, fail_fast: bool = True) -> Dict[str, List[str]]:
        """
        Validate that all AGENT_TOOL_MAPPING targets exist in the tool registry.

        This is a fail-fast check to catch misconfigurations at startup rather
        than at runtime when tasks fail.

        Args:
            fail_fast: If True, raise ToolRegistryError on first missing tool.
                      If False, collect and return all issues.

        Returns:
            Dict with 'missing' (tools in mapping but not registry) and
            'unmapped' (tools in registry but not in mapping) lists.

        Raises:
            ToolRegistryError: If fail_fast=True and validation fails.
        """
        # Get unique tools from mapping
        mapped_tools = set(AGENT_TOOL_MAPPING.values())
        registered_tools = set(self.tool_registry.keys())

        # Find missing tools (in mapping but not registered)
        missing = mapped_tools - registered_tools

        # Find unmapped tools (registered but not in mapping - just info)
        unmapped = registered_tools - mapped_tools

        issues = {
            "missing": sorted(missing),
            "unmapped": sorted(unmapped),
        }

        if missing:
            # Find which agents are affected
            affected_agents = [
                agent for agent, tool in AGENT_TOOL_MAPPING.items()
                if tool in missing
            ]

            error_msg = (
                f"Tool registry validation failed: {len(missing)} missing tools.\n"
                f"Missing tools: {sorted(missing)}\n"
                f"Affected agents: {affected_agents}\n"
                f"Ensure all MCP tools are registered before creating the executor."
            )

            logger.error(error_msg)

            if fail_fast:
                raise ToolRegistryError(error_msg)

        if unmapped:
            logger.info(
                f"Tool registry has {len(unmapped)} registered tools not in AGENT_TOOL_MAPPING: "
                f"{sorted(unmapped)}. This is informational only."
            )

        return issues

    def get_missing_tools(self) -> List[str]:
        """
        Get list of tools that are mapped but not registered.

        Returns:
            List of missing tool names.
        """
        mapped_tools = set(AGENT_TOOL_MAPPING.values())
        registered_tools = set(self.tool_registry.keys())
        return sorted(mapped_tools - registered_tools)

    async def execute_task(
        self,
        workflow_id: str,
        task_id: str,
    ) -> ExecutionResult:
        """
        Execute a pending task by invoking its mapped tool.

        Args:
            workflow_id: Parent workflow ID
            task_id: Task ID to execute

        Returns:
            ExecutionResult with status and output
        """
        start_time = datetime.now()

        # Load task from workflow state
        task = self._load_task(workflow_id, task_id)
        if not task:
            return ExecutionResult(
                task_id=task_id,
                status="ERROR",
                error=f"Task not found: {task_id}",
            )

        agent_type = task.get("agent_type", "")
        phase_name = task.get("phase", "")

        # Phase-name dispatch overrides agent-based routing: a phase in
        # ``_PHASE_TOOL_MAPPING`` reaches its dedicated handler regardless of
        # the agent_type threaded through the task. Every other phase falls
        # back to ``AGENT_TOOL_MAPPING``.
        tool_name = _PHASE_TOOL_MAPPING.get(phase_name)
        if not tool_name:
            tool_name = AGENT_TOOL_MAPPING.get(agent_type)

        if not tool_name:
            error = (
                f"No tool mapping for phase '{phase_name}' or agent "
                f"type '{agent_type}'"
            )
            logger.error(error)
            return ExecutionResult(
                task_id=task_id,
                status="ERROR",
                error=error,
            )

        # Log execution decision
        if self.capture:
            self.capture.log_decision(
                decision_type="task_execution",
                decision=f"Executing task {task_id} via tool '{tool_name}'",
                rationale=f"Agent type: {agent_type}, Workflow: {workflow_id}",
            )

        # Update task status to IN_PROGRESS
        self._update_task_status(workflow_id, task_id, "IN_PROGRESS")

        # Execute with retries
        result = await self._execute_with_retries(
            task_id=task_id,
            tool_name=tool_name,
            task_params=task,
        )

        # Calculate duration
        end_time = datetime.now()
        result.completed_at = end_time.isoformat()
        result.duration_seconds = (end_time - start_time).total_seconds()

        # Update workflow state
        self._update_task_status(
            workflow_id,
            task_id,
            result.status,
            result=result.result,
            error=result.error,
        )

        # Log completion decision. Rationale interpolates dynamic per-task
        # signals (tool, status, duration, retry count) and stays above the
        # canonical 20-char minimum so a DECISION_VALIDATION_STRICT=true run
        # does not fail-closed on the executor's own task-lifecycle capture
        # (the bare ``Duration: 0.14s`` form was 15 chars and tripped strict
        # validation, marking otherwise-successful tasks failed).
        if self.capture:
            self.capture.log_decision(
                decision_type="task_completion",
                decision=f"Task {task_id} completed with status: {result.status}",
                rationale=(
                    f"Tool '{tool_name}' for agent '{agent_type}' finished "
                    f"with status={result.status} in "
                    f"{result.duration_seconds:.2f}s "
                    f"after {result.retry_count} retry(ies)"
                ),
            )

        return result

    async def _execute_with_retries(
        self,
        task_id: str,
        tool_name: str,
        task_params: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute tool with intelligent retry logic.

        Phase 0 Hardening:
        - Uses error classification to determine retry behavior
        - Detects poison-pill patterns that should stop the batch
        - Only retries transient errors, not permanent ones
        """
        last_error = None
        error_class_value = None
        retry_count = 0
        # Log the loud GPU-OOM diagnostic at most once per task execution
        # (an OOM that persists across retry attempts would otherwise emit
        # one loud line + one VRAM probe per attempt — noise, not signal).
        _oom_logged = False

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._invoke_tool(tool_name, task_params)

                # Inspect the tool envelope for an explicit failure signal
                # before marking the task COMPLETE. Treating any parsed dict as
                # success lets ``{"success": False, ...}`` envelopes through, so
                # gate aggregation runs against an "all complete" phase summary
                # and passes gates on content that was never produced.
                #
                # ``success=False`` is a PERMANENT failure: no retry (the tool
                # already decided its own outcome), status=FAILED, with
                # error_code / error_message lifted out of the envelope so
                # downstream gate aggregation sees the failure.
                if isinstance(result, dict) and result.get("success") is False:
                    error_code = str(
                        result.get("error_code") or "TOOL_REPORTED_FAILURE"
                    )
                    error_message = str(
                        result.get("error_message")
                        or result.get("error")
                        or result.get("reason")
                        or "Tool returned success=False envelope"
                    )
                    logger.warning(
                        f"[{self.run_id}] Task {task_id} returned "
                        f"success=False envelope ({error_code}): "
                        f"{error_message}"
                    )
                    return ExecutionResult(
                        task_id=task_id,
                        status="FAILED",
                        result=result,
                        error=f"{error_code}: {error_message}",
                        error_class=error_code,
                        retry_count=retry_count,
                    )

                return ExecutionResult(
                    task_id=task_id,
                    status="COMPLETE",
                    result=result,
                    retry_count=retry_count,
                )

            except GracefulStopRequested as e:
                # Graceful stop ("checkpoint on command"): the tool observed a
                # stop sentinel at a unit boundary, checkpointed the in-flight
                # unit, and raised. This is NOT a failure — caught FIRST (before
                # asyncio.TimeoutError and the generic Exception below, even
                # though it subclasses RuntimeError) so it can NEVER be retried,
                # poison-classified, or run through the ErrorClassifier. Surface
                # a PAUSED result; ``execute_phase`` stamps the phase checkpoint
                # ``paused`` (resumable) rather than ``failed``.
                logger.info(
                    f"[{self.run_id}] Task {task_id} paused at unit boundary "
                    f"(site={getattr(e, 'site_id', '?')}, "
                    f"units_completed={getattr(e, 'units_completed', '?')})"
                )
                if self.capture:
                    self.capture.log_decision(
                        decision_type="task_execution",
                        decision=(
                            f"Task {task_id} paused on graceful-stop request"
                        ),
                        rationale=(
                            f"Graceful stop observed at site "
                            f"'{getattr(e, 'site_id', '?')}' after "
                            f"{getattr(e, 'units_completed', 0)} unit(s) via tool "
                            f"'{tool_name}'; the in-flight unit was checkpointed "
                            f"and GracefulStopRequested raised — no retry, no "
                            f"poison classification, phase stamped paused"
                        ),
                    )
                return ExecutionResult(
                    task_id=task_id,
                    status="PAUSED",
                    error=str(e),
                    error_class="paused",
                    retry_count=retry_count,
                )

            except asyncio.TimeoutError as e:
                last_error = f"Task timed out after {self.timeout_seconds}s"
                logger.warning(f"[{self.run_id}] Task {task_id} attempt {attempt + 1} timed out")

                # Phase 0: Classify timeout error
                if self.error_classifier:
                    classified = self.error_classifier.classify(e, task_id)
                    error_class_value = classified.error_class.value

                    # Check for poison pill
                    if self.poison_detector:
                        poison_result = self.poison_detector.record_failure(classified)
                        if poison_result and poison_result.triggered:
                            logger.error(f"[{self.run_id}] Poison pill detected: {poison_result.recommendation}")
                            return ExecutionResult(
                                task_id=task_id,
                                status="POISON_PILL",
                                error=f"Batch stopped: {poison_result.error_pattern}",
                                error_class="poison_pill",
                                retry_count=retry_count,
                            )

            except Exception as e:
                last_error = str(e)

                # GPU OUT-OF-MEMORY diagnostic (LOGGING ONLY — this branch
                # does NOT alter control flow). A torch CUDA OOM raised
                # during NLI/embedding scoring on a VRAM-starved box (a
                # resident local 7B holding the card) is otherwise swallowed
                # here as a bland warning. We emit a LOUD, attributable
                # diagnostic with a best-effort free-VRAM probe so the OOM is
                # greppable, then FALL THROUGH to the unchanged classifier +
                # poison-detector path below.
                #
                # Crucially we do NOT early-return a forced PERMANENT result:
                # the ``ErrorClassifier`` already matches "out of memory" via
                # its POISON_PATTERNS, so a CUDA OOM is classified POISON_PILL
                # and the existing poison-detector STOPS the batch (the
                # documented runaway-VRAM circuit breaker). An earlier version
                # short-circuited with a one-shot PERMANENT return here, which
                # silently removed that circuit breaker — the bug this redesign
                # fixes. The diagnostic is emitted at most once per task
                # execution (``_oom_logged``) to avoid repeating across retry
                # attempts.
                if _is_cuda_oom(e) and not _oom_logged:
                    _oom_logged = True
                    free_mib = _probe_free_vram_mib()
                    free_desc = (
                        f"{free_mib} MiB free"
                        if free_mib is not None
                        else "free VRAM unprobeable"
                    )
                    logger.error(
                        f"[{self.run_id}] GPU OUT OF MEMORY during task "
                        f"{task_id} (tool '{tool_name}', attempt "
                        f"{attempt + 1}): {free_desc} at failure — a "
                        f"resident model (likely the local 7B) is starving "
                        f"NLI/embedding. Deferring to the standard error "
                        f"classifier (POISON_PILL → batch circuit breaker). "
                        f"Exception: {e}"
                    )
                    if self.capture:
                        self.capture.log_decision(
                            decision_type="task_execution",
                            decision=(
                                f"Task {task_id} hit CUDA out-of-memory; emitted "
                                f"loud diagnostic, deferring to the classifier"
                            ),
                            rationale=(
                                f"GPU OOM via tool '{tool_name}' on attempt "
                                f"{attempt + 1} ({free_desc} at failure) — logged "
                                f"as a side-effect; the classify + poison-pill "
                                f"path proceeds unchanged so the runaway-VRAM "
                                f"circuit breaker fires"
                            ),
                        )

                logger.warning(f"[{self.run_id}] Task {task_id} attempt {attempt + 1} failed: {e}")

                # Phase 0: Classify error for retry decisions
                if self.error_classifier:
                    classified = self.error_classifier.classify(e, task_id)
                    error_class_value = classified.error_class.value

                    # Check for poison pill
                    if self.poison_detector:
                        poison_result = self.poison_detector.record_failure(classified)
                        if poison_result and poison_result.triggered:
                            logger.error(f"[{self.run_id}] Poison pill detected: {poison_result.recommendation}")
                            return ExecutionResult(
                                task_id=task_id,
                                status="POISON_PILL",
                                error=f"Batch stopped: {poison_result.error_pattern}",
                                error_class="poison_pill",
                                retry_count=retry_count,
                            )

                    # Don't retry permanent errors
                    if classified.error_class == ErrorClass.PERMANENT:
                        logger.info(f"[{self.run_id}] Task {task_id} has permanent error, not retrying")
                        return ExecutionResult(
                            task_id=task_id,
                            status="ERROR",
                            error=last_error,
                            error_class="permanent",
                            retry_count=retry_count,
                        )

            # Log retry decision
            retry_count += 1
            if attempt < self.max_retries and self.capture:
                rationale = f"Previous error: {last_error}"
                if error_class_value:
                    rationale += f", Error class: {error_class_value}"
                self.capture.log_decision(
                    decision_type="task_retry",
                    decision=f"Retrying task {task_id} (attempt {attempt + 2})",
                    rationale=rationale,
                )

            # Honor the configured retry backoff between attempts. Re-
            # dispatching immediately fires every retry inside a rate-limited
            # provider's cooldown window and amplifies the throttling.
            # ``RetryPolicy`` picks the curve from the ErrorClassifier's verdict
            # on the most recent failure (transient → exponential, else fixed
            # base_delay). The sleep is short-circuited under pytest so retry
            # paths don't stretch the suite by the multi-second default delay.
            if (
                attempt < self.max_retries
                and self.retry_policy
                and self.error_classifier
                and last_error is not None
                and "PYTEST_CURRENT_TEST" not in os.environ
            ):
                # Re-classify the last observed error so the policy can
                # pick the right curve. ``classify`` accepts an
                # exception OR a pre-built ClassifiedError; we pass a
                # synthetic RuntimeError carrying the message because
                # the original exception may no longer be in scope.
                classified = self.error_classifier.classify(
                    RuntimeError(last_error), task_id,
                )
                delay = self.retry_policy.get_retry_delay(attempt, classified)
                if delay > 0:
                    await asyncio.sleep(delay)

        return ExecutionResult(
            task_id=task_id,
            status="ERROR",
            error=last_error,
            error_class=error_class_value or "unknown",
            retry_count=retry_count,
        )

    async def _invoke_tool(
        self,
        tool_name: str,
        task_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke an MCP tool with properly mapped parameters.

        Uses TaskParameterMapper to translate generic task parameters
        to the tool-specific parameter names expected by each tool.

        Per-task subagent dispatch:

        When ``ED4ALL_AGENT_DISPATCH`` is truthy AND a ``dispatcher`` was
        injected AND the task's ``agent_type`` is in ``AGENT_SUBAGENT_SET``,
        the call routes through ``dispatcher.dispatch_task`` instead of
        ``tool_registry[tool_name]``. The dispatcher hands the mapped
        params to a subagent (via the mailbox bridge) that executes the
        agent's markdown spec and returns a tool-shape dict matching what
        the Python emitter would have produced. Without this fork these
        phases run as in-process templates regardless of ``--mode``.

        If any of the three conditions fail, execution falls through to
        the in-process invocation unchanged.

        Args:
            tool_name: Name of the MCP tool to invoke
            task_params: Task dict with prompt, params, context, etc.

        Returns:
            Parsed JSON result from the tool

        Raises:
            ValueError: If tool not registered
            ParameterMappingError: If required parameters are missing
        """
        # Subagent-dispatch fork: when the dispatcher + feature flag + agent
        # classification all point to subagent dispatch, route there BEFORE the
        # tool_registry lookup, so an agent with no Python tool backing it does
        # not trip the "Tool not registered" guard.
        agent_type = None
        if isinstance(task_params, dict):
            agent_type = task_params.get("agent_type")
        # Per-agent provider short-circuits: setting an agent's provider env var
        # means the operator wants their own LLM provider, so subagent dispatch
        # is bypassed and the work runs on the in-process provider lattice.
        # Agents without an env var set keep dispatching unchanged.
        #
        #   COURSEFORGE_PROVIDER        -> content-generator
        #   COURSEPLANNER_PROVIDER      -> course-outliner, via
        #       ``Courseforge.generators._outliner_provider.OutlinerProvider``
        #   TRAINFORGE_ASSESSMENT_PROVIDER -> assessment-generator, via
        #       ``Trainforge.generators._assessment_provider.AssessmentGeneratorProvider``
        #
        # Licensing is the reason these exist: synthesized LO text lands in
        # ``synthesized_objectives.json`` and propagates into every downstream
        # chunk's ``learning_outcome_refs[]``, and authored questions land in
        # ``assessments.json`` and feed ``training_synthesis`` — i.e. both
        # become training data for the resulting SLM adapter, so they must come
        # from an operator-selected license-clean provider.
        #
        # The pairs live in ``AGENT_PROVIDER_ENV_MAP`` so ``workflow_runner``
        # can emit its provider banner without restating the literals.
        _provider_env = AGENT_PROVIDER_ENV_MAP.get(agent_type or "")
        _provider_env_set = bool(
            os.environ.get(_provider_env, "").strip() if _provider_env else ""
        )
        _force_inprocess_for_courseforge = (
            _provider_env_set
            and agent_type == "content-generator"
            and _provider_env == "COURSEFORGE_PROVIDER"
        )
        _force_inprocess_for_courseplanner = (
            _provider_env_set
            and agent_type == "course-outliner"
            and _provider_env == "COURSEPLANNER_PROVIDER"
        )
        _force_inprocess_for_trainforge_assessment = (
            _provider_env_set
            and agent_type == "assessment-generator"
            and _provider_env == "TRAINFORGE_ASSESSMENT_PROVIDER"
        )
        # Wave1-I1 ToS-unblock: TRAINFORGE_SYNTHESIS_PROVIDER
        # short-circuits the Wave-74 subagent dispatch for the
        # training-synthesizer agent only. Mirrors the
        # TRAINFORGE_ASSESSMENT_PROVIDER semantics above — setting the
        # env var routes instruction-pair / preference-pair synthesis
        # through an in-process license-clean provider (local /
        # together / any registered OpenAI-compatible provider) via
        # ``Trainforge.synthesize_training.run_synthesis`` rather than
        # the Claude Code subagent. The training-pair corpus is the
        # canonical training-data surface (it literally trains the SLM
        # adapter), so per ``docs/LICENSING.md`` § "Synthesis providers"
        # Claude must NEVER author these pairs — only operator-selected
        # license-clean providers may. Closes Finding 1 of
        # ``plans/dispatch-7-execution-inspection-2026-05.md``: the
        # sibling agents (content-generator, course-outliner,
        # assessment-generator) already had provider short-circuits;
        # training-synthesizer was the sole subagent-classified agent
        # with no fail-loud guard, leaving ``--skip-training`` as the
        # only operator-discipline safety. Other Wave-74 agents keep
        # dispatching unchanged.
        _trainforge_synthesis_provider_set = bool(
            os.environ.get("TRAINFORGE_SYNTHESIS_PROVIDER", "").strip()
        )
        _force_inprocess_for_trainforge_synthesis = (
            _trainforge_synthesis_provider_set
            and agent_type == "training-synthesizer"
        )
        # trainforge_train's ``training`` phase declares the subagent-classified
        # ``training-synthesizer`` agent, but the phase-name override already
        # resolved ``tool_name`` to the real trainer. Both trainforge_train
        # handlers are deterministic compute (a trainer call, an eval harness) —
        # there is nothing for an LLM subagent to author — so keying the guard
        # on the RESOLVED tool name keeps them in-process no matter which agent
        # the YAML threads through the task. A subagent cannot fabricate an
        # adapter, so a fork here would silently produce a "trained" phase with
        # no weights.
        _force_inprocess_for_training = tool_name in _DETERMINISTIC_TRAINING_TOOLS
        if (
            _agent_dispatch_enabled()
            and self.dispatcher is not None
            and isinstance(agent_type, str)
            and agent_type in AGENT_SUBAGENT_SET
            and hasattr(self.dispatcher, "dispatch_task")
            and not _force_inprocess_for_courseforge
            and not _force_inprocess_for_courseplanner
            and not _force_inprocess_for_trainforge_assessment
            and not _force_inprocess_for_trainforge_synthesis
            and not _force_inprocess_for_training
        ):
            # Param-mapping still runs so downstream agent prompts see
            # the same shape the Python tool would have received.
            # Mapping failures surface the same way they do on the
            # legacy path (raise ParameterMappingError).
            try:
                mapped_params = self.param_mapper.map_task_to_tool_params(
                    task_params, tool_name
                )
            except ParameterMappingError as e:
                logger.error(
                    f"Parameter mapping failed for dispatch_task "
                    f"(agent={agent_type}, tool={tool_name}): {e}"
                )
                raise

            logger.info(
                f"[{self.run_id}] Routing task via dispatcher.dispatch_task "
                f"(agent={agent_type}, tool={tool_name}, "
                f"params={list(mapped_params.keys())})"
            )
            return await asyncio.wait_for(
                self.dispatcher.dispatch_task(
                    task_name=tool_name,
                    agent_type=agent_type,
                    task_params=mapped_params,
                    run_id=self.run_id,
                ),
                timeout=self.batch_timeout_seconds,
            )

        if _force_inprocess_for_courseforge:
            logger.info(
                "COURSEFORGE_PROVIDER set; bypassing content-generator "
                "subagent dispatch."
            )
        if _force_inprocess_for_courseplanner:
            logger.info(
                "COURSEPLANNER_PROVIDER set; bypassing course-outliner "
                "subagent dispatch."
            )
        if _force_inprocess_for_trainforge_assessment:
            logger.info(
                "TRAINFORGE_ASSESSMENT_PROVIDER set; bypassing "
                "assessment-generator subagent dispatch."
            )
        if _force_inprocess_for_trainforge_synthesis:
            logger.info(
                "TRAINFORGE_SYNTHESIS_PROVIDER set; bypassing "
                "training-synthesizer subagent dispatch."
            )

        # Legacy in-process path — unchanged from pre-Wave-74.
        tool_func = self.tool_registry.get(tool_name)

        if not tool_func:
            raise ValueError(f"Tool not registered: {tool_name}")

        # Use parameter mapper to get tool-specific parameters
        try:
            mapped_params = self.param_mapper.map_task_to_tool_params(
                task_params, tool_name
            )
        except ParameterMappingError as e:
            logger.error(f"Parameter mapping failed for {tool_name}: {e}")
            raise

        # Log the mapped parameters for debugging
        logger.debug(f"Invoking {tool_name} with params: {list(mapped_params.keys())}")

        # Call tool with mapped parameters under the per-task timeout.
        #
        # Task-scoped graceful stop (D10 / AMENDMENT #4). ``asyncio.wait_for``
        # would hard-cancel the tool at ``ED4ALL_TASK_TIMEOUT_MINUTES``, killing
        # an in-flight LLM call and its uncheckpointed unit. Instead we grant a
        # bounded grace window via a NON-cancelling two-stage ``asyncio.wait``,
        # signalling a per-TASK in-process stop Event (NOT the run-scoped
        # sentinel — one slow task must never pause the whole run). A stop-aware
        # in-process tool MAY consult ``current_task_stop_event`` to drain early;
        # the event is copied into the tool's task/thread context here. After the
        # grace window we hard-cancel and re-raise ``asyncio.TimeoutError`` so the
        # EXISTING TIMEOUT classification + transient-retry ladder stands
        # unchanged (a retry replays the fingerprinted resume sidecar, so no work
        # is lost). Today the channel is grace-only: no in-process tool reads the
        # event yet, so the grace window simply delays the hard cancel — but the
        # channel is wired so a stop-aware tool can shorten it later.
        stop_event = threading.Event()
        token = _TASK_STOP_EVENT.set(stop_event)
        try:
            tool_task: "asyncio.Task[Any]" = asyncio.ensure_future(
                tool_func(**mapped_params)
            )
            done, pending = await asyncio.wait(
                {tool_task}, timeout=self.timeout_seconds
            )
            if tool_task in pending:
                # Per-task deadline hit — signal the task-scoped stop channel and
                # grant the grace window. Deliberately does NOT call request_stop
                # (that would write the run-scoped sentinel and pause the run).
                stop_event.set()
                grace_seconds = _grace_seconds(self.timeout_seconds)
                logger.warning(
                    f"[{self.run_id}] Tool '{tool_name}' hit task deadline "
                    f"{self.timeout_seconds}s; signalled task-scoped stop, "
                    f"granting {grace_seconds:.1f}s grace before hard cancel"
                )
                done, pending = await asyncio.wait(
                    {tool_task}, timeout=grace_seconds
                )
            if tool_task in pending:
                # Grace elapsed — hard cancel and re-raise TimeoutError so the
                # retry ladder classifies + retries exactly as before.
                tool_task.cancel()
                try:
                    await tool_task
                except asyncio.CancelledError:
                    pass
                raise asyncio.TimeoutError(
                    f"Tool '{tool_name}' exceeded {self.timeout_seconds}s "
                    f"(+ grace) task timeout"
                )
            # Completed within the deadline or drained within grace — surface the
            # tool's own return value / exception (``.result()`` re-raises a
            # GracefulStopRequested or any tool error unchanged).
            result_str = tool_task.result()
        finally:
            _TASK_STOP_EVENT.reset(token)

        # Parse result
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            return {"raw_result": result_str}

    def _load_task(
        self,
        workflow_id: str,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load task from workflow state file."""
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return None

        try:
            with open(workflow_path) as f:
                workflow = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load workflow {workflow_id}: {e}")
            return None

        for task in workflow.get("tasks", []):
            if task.get("id") == task_id:
                return task

        return None

    def _update_task_status(
        self,
        workflow_id: str,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update task status in workflow state.

        The load+mutate+atomic-replace cycle runs under a bounded advisory
        flock so concurrent processes serialize their read-modify-write. On
        lock timeout the helper logs LOUDLY and the write still proceeds via
        atomic temp+replace — losing at worst one update, never corrupting
        the file (the 2026-07-21 interleaved-write incident).
        """
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return False

        with file_lock(
            workflow_path.with_name(workflow_path.name + ".lock"), timeout=10.0
        ):
            return self._update_task_status_locked(
                workflow_id, workflow_path, task_id, status, result, error
            )

    def _update_task_status_locked(
        self,
        workflow_id: str,
        workflow_path: Path,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Body of ``_update_task_status`` — runs under the advisory flock."""
        try:
            with open(workflow_path) as f:
                workflow = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read workflow {workflow_id} for task update: {e}")
            return False

        for task in workflow.get("tasks", []):
            if task.get("id") == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()

                if status == "IN_PROGRESS":
                    task["started_at"] = datetime.now().isoformat()
                elif status in ("COMPLETE", "ERROR", "FAILED", "TIMEOUT", "PAUSED"):
                    # PAUSED is terminal for the current run leg (the task bowed
                    # out cleanly at a unit boundary); a --resume re-dispatches
                    # it fresh from its resume sidecar.
                    task["completed_at"] = datetime.now().isoformat()

                if result is not None:
                    task["result"] = result
                if error is not None:
                    task["error"] = error

                break

        # Update progress counters
        progress = workflow.get("progress", {})
        tasks = workflow.get("tasks", [])

        progress["completed"] = sum(1 for t in tasks if t.get("status") == "COMPLETE")
        progress["in_progress"] = sum(1 for t in tasks if t.get("status") == "IN_PROGRESS")
        # Count "FAILED" and "TIMEOUT" alongside "ERROR" so persisted progress
        # reflects tool envelopes with ``success=False``, not only raised
        # exceptions.
        progress["failed"] = sum(
            1 for t in tasks
            if t.get("status") in ("ERROR", "FAILED", "TIMEOUT")
        )

        workflow["progress"] = progress
        workflow["updated_at"] = datetime.now().isoformat()

        try:
            # Atomic temp+replace — never a direct truncating write that a
            # concurrent writer could interleave with mid-document.
            atomic_write_json(workflow_path, workflow, indent=2)
            return True
        except OSError:
            return False

    async def execute_workflow(
        self,
        workflow_id: str,
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> Dict[str, ExecutionResult]:
        """
        Execute all pending tasks in a workflow.

        Args:
            workflow_id: Workflow to execute
            parallel: Run independent tasks in parallel
            max_concurrent: Max concurrent tasks

        Returns:
            Dict mapping task_id to ExecutionResult
        """
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return {}

        with open(workflow_path) as f:
            workflow = json.load(f)

        tasks = workflow.get("tasks", [])
        results = {}

        if self.capture:
            pending_count = sum(1 for t in tasks if t.get("status") == "PENDING")
            self.capture.log_decision(
                decision_type="workflow_execution",
                decision=f"Starting workflow {workflow_id} execution",
                rationale=f"Pending tasks: {pending_count}, Parallel: {parallel}",
            )

        if parallel:
            results = await self._execute_parallel(workflow_id, tasks, max_concurrent)
        else:
            results = await self._execute_sequential(workflow_id, tasks)

        return results

    def _checkpoint_task_result(
        self,
        phase_name: Optional[str],
        task_id: str,
        success: bool,
        result: Optional["ExecutionResult"] = None,
    ) -> None:
        """Record one task's outcome into the phase checkpoint as it resolves.

        Graceful-stop ("checkpoint on command") per-task ledger: called the
        instant a task finishes so a mid-phase kill leaves an accurate
        ``tasks_completed`` / ``tasks_failed`` set. Best-effort and idempotent —
        ``CheckpointManager.complete_task`` only appends a task to a list once,
        so the end-of-phase sweep re-calling it is a no-op for the ledger. A
        no-op when there is no checkpoint manager or no ``phase_name`` (the
        ``execute_workflow`` path). Artifacts are intentionally NOT passed here
        (the end-of-phase sweep carries them exactly once, avoiding a
        double-append into ``artifacts_produced``).
        """
        if not (self.checkpoint_manager and phase_name):
            return
        try:
            self.checkpoint_manager.complete_task(
                phase_name=phase_name,
                task_id=task_id,
                success=success,
            )
        except Exception as e:
            logger.warning(
                f"[{self.run_id}] Failed to checkpoint task {task_id} "
                f"in phase '{phase_name}': {e}"
            )

    async def _execute_parallel(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]],
        max_concurrent: int,
        results_sink: Optional[Dict[str, "ExecutionResult"]] = None,
    ) -> Dict[str, ExecutionResult]:
        """Execute tasks in parallel batches.

        Graceful-stop ("checkpoint on command") note: the phase this batch
        belongs to is read from ``self._active_phase_name`` (set by
        ``execute_phase`` for the duration of the call) rather than a
        parameter, so the signature stays stable for callers/tests that stub
        this method. When it is set, each task's outcome is recorded into the
        phase checkpoint the INSTANT it resolves via ``_checkpoint_task_result``
        so a mid-phase kill leaves an accurate ``tasks_completed`` /
        ``tasks_failed`` ledger; ``None`` (the ``execute_workflow`` path, which
        has no phase checkpoint) skips per-task checkpointing.

        Args:
            results_sink: optional caller-owned dict that this method
                writes each finished task's ``ExecutionResult`` into as it
                goes (in addition to returning it). ``execute_phase`` passes
                one so that when ``asyncio.wait_for`` CANCELS this coroutine
                on a whole-phase batch timeout — discarding the local return
                value — the results of tasks that ALREADY completed before
                the deadline survive in the caller's sink instead of being
                thrown away. ``None`` (the default) uses and returns a fresh
                local dict. Each task's result is recorded into the sink the
                INSTANT that task finishes (not after the whole batch's
                gather), so a batch-timeout cancellation preserves even the
                tasks that already completed within the in-flight batch — only
                the tasks still awaiting ``execute_task`` are abandoned.
        """
        # Graceful stop: the active phase (or None on the execute_workflow
        # path) is threaded via the instance attribute, not a parameter, so
        # this method's signature stays stable for callers/tests that stub it.
        phase_name = getattr(self, "_active_phase_name", None)
        results = results_sink if results_sink is not None else {}
        completed_ids = set()

        # Record each task's result into ``results`` the instant it
        # finishes rather than after the whole batch's ``gather`` returns.
        # When ``execute_phase`` wraps this coroutine in ``asyncio.wait_for``
        # and the batch deadline fires, the cancellation is delivered to the
        # in-flight ``gather``; tasks that already ran this helper to
        # completion have ALREADY written their result to the sink and are
        # preserved, while a task cancelled mid-flight raises
        # ``CancelledError`` (a BaseException, deliberately NOT caught by the
        # ``except Exception`` below) so it propagates uncaught and is left
        # unrecorded → correctly abandoned.
        async def _run_and_record(task: Dict[str, Any]) -> None:
            tid = task["id"]
            try:
                res = await self.execute_task(workflow_id, tid)
            except Exception as exc:  # noqa: BLE001 - mirror return_exceptions
                results[tid] = ExecutionResult(
                    task_id=tid,
                    status="ERROR",
                    error=str(exc),
                )
                task["status"] = "ERROR"
                # Graceful-stop per-task ledger: record the failure the instant
                # it resolves so a mid-phase kill sees it (ledger only, no
                # artifacts — the end-of-phase sweep carries artifacts once).
                self._checkpoint_task_result(phase_name, tid, success=False)
                return
            results[tid] = res
            if res.status == "COMPLETE":
                completed_ids.add(tid)
            task["status"] = res.status
            # Per-task-as-completed checkpointing (graceful stop): stamp the
            # ledger the INSTANT each task resolves so a mid-phase kill leaves an
            # accurate tasks_completed/tasks_failed set. PAUSED tasks are LEFT in
            # tasks_pending (a --resume re-runs them from their resume sidecar).
            # Idempotent with the end-of-phase complete_task sweep in
            # execute_phase (that sweep is what carries artifacts, exactly once).
            if res.status != "PAUSED":
                self._checkpoint_task_result(
                    phase_name, tid, success=res.status == "COMPLETE"
                )

        # ROLLING CONCURRENCY WINDOW (replaces the former batch barrier).
        #
        # The old loop sliced the dependency-satisfied frontier to
        # ``max_concurrent`` and blocked on ``asyncio.gather`` over the WHOLE
        # slice before dispatching the next slice. Concurrency therefore
        # started at ``max_concurrent`` and DRAINED toward 1 as the fast tasks
        # in a slice finished while gather waited on the slowest — the exact
        # GPU-starving anti-pattern. This version keeps up to
        # ``max_concurrent`` tasks IN FLIGHT and refills a freed slot from the
        # frontier the INSTANT any single task completes, so the pipe stays
        # full at ``max_concurrent`` for the whole phase. Every contract the
        # barrier upheld is preserved (see the per-block comments):
        #   * dependency DAG gating — refill only from deps-satisfied tasks;
        #   * no mid-flight cancellation of a STARTED task on stop/poison —
        #     we stop DISPATCHING and drain the in-flight set;
        #   * poison-pill halt — checked per completion (finer than the batch
        #     boundary), halts dispatch, drains in-flight, same result shape;
        #   * stop-sentinel — probed before each new dispatch, drains in-flight
        #     to their unit-boundary checkpoint, marks un-run PENDING PAUSED;
        #   * per-task checkpoint — unchanged (written inside _run_and_record);
        #   * whole-phase wall bound — the ``execute_phase`` batch-timeout
        #     wrapper still wraps THIS coroutine unchanged, so a wedged task
        #     cannot hang the phase past the deadline + grace (a rolling loop
        #     is strictly friendlier: fast tasks keep completing + checkpointing
        #     right up to the deadline instead of stalling at a drained batch
        #     boundary).
        # ``max_concurrent`` (the width) is untouched — rolling keeps the pipe
        # full at exactly today's ceiling, it does NOT raise it.

        # in-flight asyncio.Task -> the task dict it is executing.
        inflight: Dict["asyncio.Task[None]", Dict[str, Any]] = {}
        dispatched_ids: set = set()  # task ids already handed to _run_and_record
        halt = False  # tripped by stop-sentinel or poison-pill: dispatch no more

        def _ready_frontier() -> List[Dict[str, Any]]:
            """PENDING, not-yet-dispatched tasks whose deps are all COMPLETE.

            Mirrors the barrier's frontier scan exactly: already-COMPLETE tasks
            (e.g. from a resume) fold into ``completed_ids`` so their dependents
            unlock, and a task only becomes runnable when EVERY dependency is in
            ``completed_ids`` — a task completing may unlock new dependents,
            which the next call surfaces.
            """
            ready: List[Dict[str, Any]] = []
            for task in tasks:
                tid = task.get("id")
                status = task.get("status")
                if status == "COMPLETE":
                    completed_ids.add(tid)
                    continue
                if status != "PENDING":
                    continue
                if tid in dispatched_ids:
                    continue  # in flight or finished — never double-dispatch
                deps = task.get("dependencies", [])
                if all(d in completed_ids for d in deps):
                    ready.append(task)
            return ready

        while True:
            # Graceful stop ("checkpoint on command"): probe the sentinel
            # BEFORE dispatching anything new (mirrors the barrier's loop-top
            # check + the ED4ALL_NLI_CROSSBLOCK template's per-submit poll).
            # Once observed we STOP DISPATCHING but let in-flight tasks drain to
            # their next unit boundary + checkpoint — never cancel a started
            # request (that would risk a partial/corrupt artifact). Un-run
            # PENDING tasks are marked PAUSED after the drain (below).
            if not halt and stop_requested(self.run_id):
                halt = True
                logger.info(
                    f"[{self.run_id}] Graceful stop observed"
                    + (f" in phase '{phase_name}'" if phase_name else "")
                    + "; halting dispatch (in-flight tasks draining to a unit "
                    "boundary, no new tasks dispatched)"
                )

            # Refill free slots from the dependency-satisfied frontier. Only
            # while NOT halted — a stop/poison halt drains but never dispatches.
            if not halt:
                while len(inflight) < max_concurrent:
                    frontier = _ready_frontier()
                    if not frontier:
                        break
                    task = frontier[0]
                    dispatched_ids.add(task["id"])
                    fut = asyncio.ensure_future(_run_and_record(task))
                    inflight[fut] = task

            # Nothing in flight and nothing left to dispatch → phase done.
            if not inflight:
                break

            # Wait for the FIRST in-flight task to finish (not the whole
            # slice). This is both the steady-state refill trigger AND the
            # drain: on halt we add no new work but still await every in-flight
            # task so it checkpoints.
            try:
                done, _pending = await asyncio.wait(
                    set(inflight), return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                # This coroutine was CANCELLED — which happens ONLY on the
                # ``execute_phase`` batch-timeout HARD kill (the grace window
                # already expired with a worker unresponsive to the sentinel).
                # Mirror ``asyncio.gather``'s cancellation propagation that the
                # barrier relied on: cancel the in-flight children so their
                # inner ``execute_task`` raises CancelledError and is left
                # unrecorded → correctly abandoned (the results_sink already
                # holds every task that finished before the deadline). This is
                # the ONLY place a started task is cancelled, and it is the
                # pre-existing hard-timeout behaviour — the graceful-stop and
                # poison-pill paths above set ``halt`` and DRAIN instead.
                for fut in inflight:
                    fut.cancel()
                raise

            for fut in done:
                finished_task = inflight.pop(fut)
                if fut.cancelled():  # defensive — not reached on a normal wait
                    continue
                # ``_run_and_record`` never raises (it funnels every Exception
                # into an ERROR ExecutionResult); surface anything unexpected
                # rather than swallow it.
                exc = fut.exception()
                if exc is not None:
                    raise exc

                # Poison-pill: check the INSTANT this task resolves (finer-
                # grained than the batch boundary — a strict improvement). Once
                # tripped, dispatch no more; the loop then drains the remaining
                # in-flight tasks and halts with the same result shape the
                # barrier produced. In-flight siblings are NOT cancelled (same
                # anti-partial-artifact guarantee).
                tid = finished_task.get("id")
                r = results.get(tid)
                if (
                    not halt
                    and r is not None
                    and not isinstance(r, Exception)
                    and getattr(r, "status", None) == "POISON_PILL"
                ):
                    halt = True
                    logger.error(
                        f"[{self.run_id}] Poison pill observed; halting dispatch "
                        f"(in-flight tasks draining, remaining runnables skipped)"
                    )

        # After the loop: if we halted on a STOP sentinel (not poison — poison
        # writes no sentinel), mark still-PENDING, never-dispatched tasks PAUSED
        # so ``execute_phase`` detects the stop and stamps the phase checkpoint
        # ``paused`` rather than reporting success on a partially-run phase;
        # they stay in the checkpoint's tasks_pending for --resume. Idempotent
        # with the pre-armed-sentinel case (nothing was ever dispatched).
        if stop_requested(self.run_id):
            for task in tasks:
                tid = task.get("id")
                if task.get("status") == "PENDING" and tid not in results:
                    results[tid] = ExecutionResult(
                        task_id=tid,
                        status="PAUSED",
                        error="Graceful stop requested; task not dispatched",
                        error_class="paused",
                    )
                    task["status"] = "PAUSED"

        return results

    async def _execute_sequential(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, ExecutionResult]:
        """Execute tasks sequentially."""
        results = {}

        for task in tasks:
            if task.get("status") != "PENDING":
                continue

            # Graceful stop ("checkpoint on command"): mirror the parallel
            # loop-top check — before dispatching the next task, if a stop
            # sentinel is present, mark the remaining PENDING tasks PAUSED and
            # halt (never mid-task; the previous task already completed).
            if stop_requested(self.run_id):
                for remaining in tasks:
                    rid = remaining.get("id")
                    if remaining.get("status") == "PENDING" and rid not in results:
                        results[rid] = ExecutionResult(
                            task_id=rid,
                            status="PAUSED",
                            error="Graceful stop requested; task not dispatched",
                            error_class="paused",
                        )
                        remaining["status"] = "PAUSED"
                logger.info(
                    f"[{self.run_id}] Graceful stop observed; halting "
                    f"sequential execution (no new tasks dispatched)"
                )
                break

            result = await self.execute_task(workflow_id, task["id"])
            results[task["id"]] = result

            if result.status == "ERROR":
                # Stop on error in sequential mode
                break

        return results

    def register_tool(
        self,
        tool_name: str,
        tool_func: Callable[..., Awaitable[str]],
    ) -> None:
        """Register a tool function for execution."""
        self.tool_registry[tool_name] = tool_func

    # =========================================================================
    # Phase 0 Hardening: Phase Execution with Checkpoints and Validation Gates
    # =========================================================================

    async def execute_phase(
        self,
        workflow_id: str,
        phase_name: str,
        phase_index: int,
        tasks: List[Dict[str, Any]],
        gate_configs: Optional[List[Dict[str, Any]]] = None,
        max_concurrent: int = 5,
        phase_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
        workflow_params: Optional[Dict[str, Any]] = None,
        extract_phase_outputs_fn: Optional[
            Callable[[str, Dict[str, "ExecutionResult"]], Dict[str, Any]]
        ] = None,
        phase_batch_timeout_minutes: Optional[int] = None,
    ) -> Tuple[Dict[str, ExecutionResult], bool, Optional[List[Dict]]]:
        """
        Execute a workflow phase with checkpointing and validation gates.

        Phase 0 Hardening:
        - Creates checkpoint at phase start
        - Updates checkpoint after each task
        - Runs validation gates at phase end
        - Supports crash recovery via checkpoints

        Args:
            workflow_id: Parent workflow ID
            phase_name: Name of the phase
            phase_index: Index of phase in workflow
            tasks: List of tasks to execute
            gate_configs: Optional list of validation gate configurations
            max_concurrent: Maximum concurrent tasks
            phase_batch_timeout_minutes: Optional per-phase whole-batch
                wall-clock timeout (minutes) sourced from the phase config
                (workflows.yaml ``batch_timeout_minutes``). When set and
                positive it OVERRIDES the executor-wide
                ``self.batch_timeout_seconds`` fallback for THIS phase only,
                so a phase that declares ``batch_timeout_minutes: 240`` gets
                14400s even when ``ED4ALL_BATCH_TIMEOUT_MINUTES`` is unset.
                ``None`` (no YAML value) preserves the prior env/30-min
                fallback exactly.

        Returns:
            Tuple of (results dict, gates_passed bool, gate_results list)
        """
        task_ids = [t.get("id") for t in tasks]
        gate_results = None

        # Resolve the effective whole-phase batch timeout. Precedence:
        # per-phase YAML ``batch_timeout_minutes`` (if set & positive) →
        # ``self.batch_timeout_seconds`` (which already resolved
        # ``ED4ALL_BATCH_TIMEOUT_MINUTES`` env → 30 at construction). This is
        # the plumbing that makes workflows.yaml's per-phase value live.
        batch_timeout_seconds = self.batch_timeout_seconds
        if phase_batch_timeout_minutes is not None:
            try:
                _phase_min = int(phase_batch_timeout_minutes)
                if _phase_min > 0:
                    batch_timeout_seconds = _phase_min * 60
            except (TypeError, ValueError):
                pass

        # W2: clear cross-phase poison-pill state at every phase boundary.
        # The poison detector accumulates errors keyed by pattern hash for
        # the lifetime of the executor; without this reset, three same-
        # pattern errors spread across phases N, N+1, ... falsely trip the
        # detector on phase N+1's first error. Per-phase reset matches the
        # contract that poison detection halts a batch, not the whole workflow.
        self.reset_poison_detector()

        # Start checkpoint
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.start_phase(
                    run_id=self.run_id,
                    workflow_id=workflow_id,
                    phase_name=phase_name,
                    phase_index=phase_index,
                    task_ids=task_ids
                )
                logger.info(f"[{self.run_id}] Started phase checkpoint: {phase_name}")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to create phase checkpoint: {e}")

        # Log phase start
        if self.capture:
            self.capture.log_decision(
                decision_type="phase_start",
                decision=f"Starting phase: {phase_name}",
                rationale=f"Phase {phase_index}, {len(tasks)} tasks, max_concurrent={max_concurrent}",
            )

        # Execute tasks with batch timeout.
        #
        # A whole-phase batch timeout must NOT discard the results (and
        # checkpoints) of tasks that ALREADY completed before the deadline.
        # ``asyncio.wait_for`` cancels ``_execute_parallel`` and throws away its
        # local return value, so rebuilding ``results`` in the ``except`` arm
        # loses every finished task's real ExecutionResult. Thread a caller-
        # owned ``results_sink`` instead so completed results survive the
        # cancellation; on timeout preserve those and mark only the
        # still-unfinished tasks TIMEOUT (abandoned / retried later).
        partial_results: Dict[str, ExecutionResult] = {}
        # Graceful stop: publish the active phase for ``_execute_parallel`` /
        # ``_run_and_record`` (they read ``self._active_phase_name`` rather than
        # take a new parameter, keeping the signature stubbable). Restored in
        # ``finally`` so a nested/next phase never inherits a stale name.
        _prev_active_phase = getattr(self, "_active_phase_name", None)
        self._active_phase_name = phase_name
        # Metering context: also publish the active phase in the environment so
        # an in-process content-gen LLM usage tap (which is NOT passed the
        # executor instance) can stamp the SPENDING phase on its
        # ``llm_usage.jsonl`` row — the local OpenAI-compatible seat serves many
        # phases and cannot know its phase from a static literal the way the
        # SemantiK cascade taps do. Env name kept in sync with the tap's
        # ``Trainforge.generators._openai_compatible_client.ENV_ACTIVE_PHASE``.
        # Restored in ``finally`` so a nested / next phase never inherits a
        # stale value; best-effort (never perturbs phase execution).
        _active_phase_env = "ED4ALL_ACTIVE_PHASE"
        _prev_active_phase_env = os.environ.get(_active_phase_env)
        try:
            if phase_name:
                os.environ[_active_phase_env] = str(phase_name)
        except Exception:  # noqa: BLE001 — metering context must never block a phase
            pass
        try:
            # Graceful-stop timeout → pause. ``asyncio.wait_for``
            # CANCELS ``_execute_parallel`` at the deadline — precisely the hard
            # kill the "checkpoint on command" grace period must prevent. So we
            # wrap the batch in a Task and enforce the deadline with a
            # NON-cancelling ``asyncio.wait``, in two stages:
            #   stage 1 — wait to the batch deadline; on expiry request a
            #     RUN-SCOPED graceful stop (the whole-phase pause path — a
            #     timeout becomes a pause, never a failure), then
            #   stage 2 — wait a bounded grace window for the in-flight workers
            #     to reach their next unit boundary. Each worker's cooperative
            #     in-loop stop-check observes the sentinel,
            #     checkpoints its in-flight unit, and returns a PAUSED result;
            #     the batch loop-top halts new dispatch. A grace-drained batch
            #     therefore surfaces PAUSED results, NOT TIMEOUT.
            # Only if the grace window ALSO expires (an unresponsive worker that
            # never consults the sentinel) do we hard-cancel and fall through to
            # the pre-existing TIMEOUT marking + retry.
            batch_task: "asyncio.Task[Dict[str, ExecutionResult]]" = (
                asyncio.ensure_future(
                    self._execute_parallel(
                        workflow_id, tasks, max_concurrent,
                        results_sink=partial_results,
                    )
                )
            )
            done, pending = await asyncio.wait(
                {batch_task}, timeout=batch_timeout_seconds
            )
            if batch_task in pending:
                # Deadline hit with work still in flight — convert the timeout
                # into a run-scoped graceful stop (NOT a cancel) so the phase
                # pauses instead of losing in-flight units.
                request_stop(
                    self.run_id,
                    scope="run",
                    reason="batch_timeout",
                    source="timeout",
                )
                grace_seconds = _grace_seconds(batch_timeout_seconds)
                logger.warning(
                    f"[{self.run_id}] Phase {phase_name} hit batch deadline "
                    f"{batch_timeout_seconds}s; requested graceful stop, "
                    f"granting {grace_seconds:.1f}s grace for in-flight workers "
                    f"to drain to a unit boundary before hard cancel"
                )
                done, pending = await asyncio.wait(
                    {batch_task}, timeout=grace_seconds
                )

            if batch_task in pending:
                # Grace ALSO expired — a worker never reached a unit boundary.
                # Restore the pre-graceful HARD behaviour: clear the
                # timeout-authored sentinel (it was written by the timeout
                # machinery, not the operator — leaving it would spuriously
                # pause the NEXT phase of this run), hard-cancel, and TIMEOUT-mark
                # only the still-unfinished tasks exactly as before. Completed
                # results survive in the ``partial_results`` sink.
                clear_stop(self.run_id)
                batch_task.cancel()
                try:
                    await batch_task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 - coroutine's own error, if any
                    pass
                results = dict(partial_results)
                timed_out_ids: List[Any] = []
                for t in tasks:
                    tid = t.get("id")
                    if tid in results:
                        continue  # finished before the deadline — preserve it
                    if t.get("status") != "PENDING":
                        continue  # not part of this phase's run
                    results[tid] = ExecutionResult(
                        task_id=tid,
                        status="TIMEOUT",
                        error=f"Phase batch timeout after {batch_timeout_seconds}s",
                    )
                    timed_out_ids.append(tid)
                logger.error(
                    f"[{self.run_id}] Phase {phase_name} timed out after "
                    f"{batch_timeout_seconds}s (grace {grace_seconds:.1f}s "
                    f"elapsed with a worker still unresponsive) — preserved "
                    f"{len(partial_results)} completed task result(s)/"
                    f"checkpoint(s); marked {len(timed_out_ids)} unfinished "
                    f"task(s) TIMEOUT"
                )
                if self.capture:
                    self.capture.log_decision(
                        decision_type="phase_completion",
                        decision=(
                            f"Phase {phase_name} batch timeout: preserved "
                            f"{len(partial_results)} completed, abandoned "
                            f"{len(timed_out_ids)} unfinished"
                        ),
                        rationale=(
                            f"Batch wall-clock deadline {batch_timeout_seconds}s "
                            f"then grace {grace_seconds:.1f}s both elapsed on "
                            f"phase {phase_name} (index {phase_index}) with a "
                            f"worker unresponsive to the stop sentinel; completed "
                            f"task results + their checkpoints preserved via the "
                            f"results_sink, only the {len(timed_out_ids)} "
                            f"unfinished task(s) abandoned for retry "
                            f"(ids={timed_out_ids})"
                        ),
                    )
            else:
                # Batch finished — either fully before the deadline, or drained
                # to a COMPLETE/PAUSED mix within the grace window. The
                # ``paused_detected`` path below stamps the phase ``paused``
                # whenever any worker drained on the graceful stop.
                results = batch_task.result()
        finally:
            # Graceful stop: clear the published active phase so a subsequent
            # (or nested) phase never inherits a stale name. Per-task
            # checkpointing only needs it while ``_execute_parallel`` ran above.
            self._active_phase_name = _prev_active_phase
            # Restore the published active-phase metering env to its prior value
            # (or remove it when it was unset) so a subsequent / nested phase
            # never stamps usage rows with a stale phase. Best-effort.
            try:
                if _prev_active_phase_env is None:
                    os.environ.pop(_active_phase_env, None)
                else:
                    os.environ[_active_phase_env] = _prev_active_phase_env
            except Exception:  # noqa: BLE001 — metering context cleanup must never raise
                pass

        # Update checkpoint with task results. This end-of-phase sweep is the
        # idempotent complement to the per-task ``_checkpoint_task_result``
        # ledger written during ``_execute_parallel``: it is the pass that
        # carries artifacts into the checkpoint (exactly once), while
        # ``complete_task``'s "append only if absent" guard keeps the ledger
        # itself idempotent. PAUSED tasks are SKIPPED — a graceful-stop task is
        # neither completed nor failed; leaving it out keeps it in the
        # checkpoint's ``tasks_pending`` for a --resume.
        if self.checkpoint_manager:
            for task_id, result in results.items():
                if result.status == "PAUSED":
                    continue
                try:
                    artifacts = result.artifacts if hasattr(result, 'artifacts') else []
                    self.checkpoint_manager.complete_task(
                        phase_name=phase_name,
                        task_id=task_id,
                        success=result.status == "COMPLETE",
                        artifacts=[a for a in artifacts] if artifacts else None
                    )
                except Exception as e:
                    logger.warning(f"[{self.run_id}] Failed to update task checkpoint: {e}")

        # Check for poison pill status
        poison_detected = any(r.status == "POISON_PILL" for r in results.values())
        if poison_detected:
            logger.error(f"[{self.run_id}] Phase {phase_name} stopped due to poison pill")
            if self.checkpoint_manager:
                self.checkpoint_manager.fail_phase(phase_name, "Poison pill detected")
            return results, False, None

        # Graceful stop ("checkpoint on command"): any PAUSED task result means
        # the phase stopped at a unit boundary. Stamp the phase checkpoint
        # ``paused`` (resumable, NEVER failed), SKIP the validation gates and
        # ``fail_phase``, and return. The paused signal is surfaced to the
        # caller (WorkflowRunner) via the presence of ``status == "PAUSED"``
        # ExecutionResults in the returned ``results`` dict — the least-invasive
        # widening (no signature change). ``gates_passed`` is returned True and
        # ``gate_results`` None so the runner does NOT treat a pause as a gate
        # failure; the runner detects the pause from the results and halts
        # downstream phases, stamping the workflow PAUSED.
        paused_detected = any(r.status == "PAUSED" for r in results.values())
        if paused_detected:
            n_paused = sum(1 for r in results.values() if r.status == "PAUSED")
            logger.info(
                f"[{self.run_id}] Phase {phase_name} paused (graceful stop) — "
                f"{n_paused} task(s) left pending for --resume"
            )
            if self.checkpoint_manager:
                try:
                    self.checkpoint_manager.pause_phase(
                        phase_name, reason="Graceful stop requested"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.run_id}] Failed to stamp phase checkpoint "
                        f"paused for {phase_name}: {e}"
                    )
            if self.capture:
                self.capture.log_decision(
                    decision_type="phase_completion",
                    decision=f"Phase {phase_name} paused on graceful-stop request",
                    rationale=(
                        f"Phase {phase_name} (index {phase_index}) observed a stop "
                        f"sentinel at a unit boundary; {n_paused} task(s) left "
                        f"PENDING for --resume, validation gates skipped, "
                        f"checkpoint stamped paused (resumable, not failed)"
                    ),
                )
            return results, True, None

        # Run validation gates (per-gate input routing)
        gates_passed = True
        if gate_configs and self.gate_manager and HARDENING_VALIDATION_GATES:
            # Build the fallback artifacts blob for validators not yet in
            # the router registry (legacy / unknown paths).
            all_artifacts = []
            for result in results.values():
                if hasattr(result, 'artifacts') and result.artifacts:
                    all_artifacts.extend(result.artifacts)
                if result.result and isinstance(result.result, dict):
                    if 'artifacts' in result.result:
                        all_artifacts.extend(result.result['artifacts'])
            fallback_inputs = {'artifacts': all_artifacts, 'results': results}

            # Accumulated phase outputs + workflow params feed the router.
            # Callers (WorkflowRunner) pass these explicitly; legacy
            # callers that don't get an empty blob → every gate without
            # a builder route falls back to fallback_inputs.
            _phase_outputs = dict(phase_outputs or {})
            _workflow_params = workflow_params or {}

            # Extract the current phase's outputs into ``_phase_outputs``
            # BEFORE running the gate router, so builders can resolve inputs
            # produced by THIS phase. ``WorkflowRunner.run_workflow`` only calls
            # ``_extract_phase_outputs`` AFTER ``execute_phase`` returns, so
            # without this injection the router sees prior phases only and every
            # gate whose inputs come from the phase it guards logs
            # "skipped — missing inputs". This gives the router one source of
            # truth: all outputs up to and including the in-progress phase.
            if extract_phase_outputs_fn is not None:
                try:
                    current_extracted = extract_phase_outputs_fn(
                        phase_name, results,
                    )
                    if isinstance(current_extracted, dict) and current_extracted:
                        # Merge into a phase-indexed block (same shape
                        # as prior phase_outputs entries) AND surface
                        # the same keys at the top level so builders
                        # that lookup `phase_outputs[phase_name][key]`
                        # AND builders that lookup by key across all
                        # phases both resolve cleanly.
                        merged_phase_block = dict(
                            _phase_outputs.get(phase_name, {})
                        )
                        merged_phase_block.update(current_extracted)
                        _phase_outputs[phase_name] = merged_phase_block
                except Exception as exc:
                    logger.warning(
                        f"[{self.run_id}] Failed to extract current-phase "
                        f"outputs for gate routing on {phase_name}: {exc}"
                    )

            gate_results_list = []
            parsed_gates = []
            # gate_id -> declared severity string. ``GateResult`` carries NO
            # severity field, so unless the executor stamps the parsed
            # ``GateConfig`` severity onto each persisted/returned result dict,
            # a post-hoc reader cannot tell a blocking (critical) failure from
            # an advisory (warning) one — and an all-warning failure set reads
            # as an unexplained phase-level gates_passed=False.
            severity_by_gate_id: Dict[str, str] = {}
            for gc in gate_configs:
                try:
                    if isinstance(gc, GateConfig):
                        gate = gc
                    else:
                        # Loud, non-fatal signal when a gate omits an
                        # explicit severity: the fail-closed default is
                        # CRITICAL, so a silent omission BLOCKS the phase.
                        # Surfacing it here keeps the "why did a phase
                        # block?" diagnosis honest.
                        if 'severity' not in gc:
                            logger.warning(
                                f"[{self.run_id}] Gate "
                                f"{gc.get('gate_id', 'unknown')} declares no "
                                "'severity' — defaulting to CRITICAL "
                                "(fail-closed, so a failure will BLOCK). "
                                "Set 'severity' explicitly in workflows.yaml "
                                "to silence this."
                            )
                        # Parse via the canonical ``from_dict`` so the YAML
                        # ``behavior:`` block (on_fail / on_error) reaches
                        # GateConfig; constructing GateConfig by hand drops it
                        # and silently defaults to BLOCK / FAIL_CLOSED. Copy
                        # first: ``from_dict`` POPS ``behavior`` and would
                        # otherwise mutate the shared
                        # ``phase.validation_gates`` dict. Normalise the
                        # ``validator``/``validator_path`` alias before
                        # delegating.
                        gc2 = dict(gc)
                        gc2.setdefault(
                            'validator', gc2.get('validator_path', ''),
                        )
                        gate = GateConfig.from_dict(gc2)
                    parsed_gates.append(gate)
                    severity_by_gate_id[gate.gate_id] = gate.severity.value
                except Exception as e:
                    logger.warning(f"[{self.run_id}] Invalid gate config: {e}")

            # Opt-in shared per-block feature cache
            # (ED4ALL_VALIDATION_FEATURE_CACHE, default OFF). ONE instance per
            # gate-chain invocation, threaded into every builder (cache=) and
            # every validator (merged_inputs["feature_cache"]) so the ~424-block
            # hydration, chunks.jsonl parse, HTML strip, sentence split, and
            # embed work is computed once and shared across the whole gate suite
            # instead of re-run per gate. Default OFF → never constructed →
            # every seam sees cache=None → byte-identical. Never let a cache
            # construction failure block the phase.
            feature_cache = None
            try:
                from lib.validators.feature_cache import (
                    BlockFeatureCache,
                    resolve_feature_cache_enabled,
                )

                if resolve_feature_cache_enabled():
                    feature_cache = BlockFeatureCache(
                        _phase_outputs, _workflow_params,
                    )
            except Exception as exc:  # noqa: BLE001 — cache is best-effort
                logger.warning(
                    f"[{self.run_id}] BlockFeatureCache construction failed "
                    f"({exc}); gates fall back to self-compute."
                )
                feature_cache = None

            for gate in parsed_gates:
                # Per-gate input build.
                inputs: Dict[str, Any]
                missing: List[str] = []
                if self.gate_input_router is not None and gate.validator_path:
                    if feature_cache is not None:
                        # Opt-in path only: pass the cache kwarg exclusively when
                        # a cache was constructed, so a custom router whose build()
                        # predates the kwarg keeps the exact legacy signature when
                        # the flag is off (byte-identical).
                        inputs, missing = self.gate_input_router.build(
                            gate.validator_path, _phase_outputs, _workflow_params,
                            cache=feature_cache,
                        )
                    else:
                        inputs, missing = self.gate_input_router.build(
                            gate.validator_path, _phase_outputs, _workflow_params,
                        )
                else:
                    inputs = dict(fallback_inputs)

                # If the builder flagged missing required inputs, mark
                # the gate as skipped rather than silently passing.
                if missing:
                    reason = ", ".join(missing)
                    logger.warning(
                        f"[{self.run_id}] Gate {gate.gate_id} "
                        f"({gate.validator_path}) skipped — missing inputs: "
                        f"{reason}"
                    )
                    skipped_result = GateResult(
                        gate_id=gate.gate_id,
                        validator_name=gate.validator_path,
                        validator_version="skipped",
                        passed=True,
                        score=None,
                        issues=[GateIssue(
                            severity="warning",
                            code="GATE_SKIPPED_MISSING_INPUTS",
                            message=(
                                f"Gate skipped: builder could not resolve "
                                f"required inputs ({reason}). This is a "
                                "structured skip, not a silent pass — the "
                                "gate did not run."
                            ),
                            suggestion=(
                                "Ensure the phase's upstream outputs "
                                "surface the required keys, or add a "
                                "builder for this validator in "
                                "MCP/hardening/gate_input_routing.py."
                            ),
                        )],
                    )
                    # Mark as skipped in a forward-compat way.
                    try:
                        skipped_result.waiver_info = {"skipped": "true", "reason": reason}
                    except Exception:
                        pass
                    gate_results_list.append(skipped_result)
                    continue

                # Merge the router-produced inputs with fallback blob
                # under non-colliding keys so legacy validators that
                # look for 'artifacts' still find it.
                merged_inputs: Dict[str, Any] = dict(fallback_inputs)
                merged_inputs.update(inputs)

                # H3 Worker S0.5: inject the executor's capture so any
                # validator that reads ``inputs.get("decision_capture")``
                # (Pattern A) or ``inputs.get("capture")`` (Pattern B —
                # family_completeness, eval_gating) actually receives a
                # live capture in production. ``setdefault`` so an
                # explicit per-builder override still wins.
                if self.capture is not None:
                    merged_inputs.setdefault("decision_capture", self.capture)
                    merged_inputs.setdefault("capture", self.capture)

                # Inject the shared feature cache (opt-in;
                # ED4ALL_VALIDATION_FEATURE_CACHE). ``setdefault`` so an explicit
                # per-builder override still wins; absent → validators self-compute.
                if feature_cache is not None:
                    merged_inputs.setdefault("feature_cache", feature_cache)

                # Run the gate via the manager (handles waivers + errors)
                _gate_t0 = time.monotonic()
                result = self.gate_manager.run_gate(gate, merged_inputs)
                logger.info(
                    f"[{self.run_id}] gate {gate.gate_id} finished in "
                    f"{time.monotonic() - _gate_t0:.1f}s (passed={result.passed})"
                )
                gate_results_list.append(result)

                # Honour severity / behavior-on-fail for gate ordering.
                if not result.passed:
                    if gate.severity == GateSeverity.CRITICAL:
                        gates_passed = False

            # NOTE (no truncation / no cap): every parsed gate that ran
            # (or was structured-skipped) contributes exactly one entry
            # here — ``len(gate_results) == len(parsed_gates)``. There is
            # no 50-entry cap or merge; a persisted checkpoint with N
            # entries reflects N configured gates, nothing hidden.
            gate_results = []
            for gr in gate_results_list:
                d = gr.to_dict() if hasattr(gr, 'to_dict') else dict(gr)
                # Stamp the declared gate severity onto each result so the
                # persisted checkpoint + the returned chain stay
                # severity-auditable (warning vs critical) long after the
                # run. GateResult itself has no severity field.
                if isinstance(d, dict) and d.get('severity') is None:
                    sev = severity_by_gate_id.get(d.get('gate_id'))
                    if sev is not None:
                        d['severity'] = sev
                gate_results.append(d)

            # Log gate results
            if self.capture:
                for gr in gate_results_list:
                    skipped = bool(getattr(gr, 'waiver_info', None) and isinstance(gr.waiver_info, dict) and gr.waiver_info.get('skipped') == 'true')
                    if skipped:
                        status = "SKIPPED"
                    else:
                        status = "PASSED" if gr.passed else "FAILED"
                    self.capture.log_decision(
                        decision_type="validation_result",
                        decision=f"Gate {gr.gate_id}: {status}",
                        rationale=f"Score: {gr.score}, Issues: {len(gr.issues)}",
                    )

        # Complete or fail checkpoint
        if self.checkpoint_manager:
            try:
                if gates_passed:
                    validation_results = {'gate_results': gate_results} if gate_results else {}
                    self.checkpoint_manager.complete_phase(phase_name, validation_results)
                    logger.info(f"[{self.run_id}] Completed phase checkpoint: {phase_name}")
                else:
                    self.checkpoint_manager.fail_phase(
                        phase_name,
                        "Validation gates failed",
                        validation_results=(
                            {'gate_results': gate_results} if gate_results else None
                        ),
                    )
                    logger.warning(f"[{self.run_id}] Phase {phase_name} failed validation gates")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to finalize phase checkpoint: {e}")

        # Log phase completion
        if self.capture:
            completed = sum(1 for r in results.values() if r.status == "COMPLETE")
            # Include FAILED so task envelopes with ``success=False`` surface in
            # the phase summary rather than being lumped under "completed".
            failed = sum(
                1 for r in results.values()
                if r.status in ("ERROR", "TIMEOUT", "FAILED")
            )
            self.capture.log_decision(
                decision_type="phase_completion",
                decision=f"Phase {phase_name} completed: {completed} success, {failed} failed",
                rationale=(
                    f"Phase {phase_name} (index {phase_index}) finished with "
                    f"{completed}/{len(tasks)} tasks succeeding, {failed} "
                    f"failed; gates_passed={gates_passed}, "
                    f"max_concurrent={max_concurrent}"
                ),
            )

        return results, gates_passed, gate_results

    def get_resumable_phase(self) -> Optional[Dict[str, Any]]:
        """
        Check for incomplete phases that can be resumed.

        Returns:
            Phase checkpoint dict if resumable phase exists, None otherwise
        """
        if not self.checkpoint_manager:
            return None

        checkpoint = self.checkpoint_manager.get_resumable_phase()
        if checkpoint:
            return {
                'phase_name': checkpoint.phase_name,
                'phase_index': checkpoint.phase_index,
                'tasks_completed': checkpoint.tasks_completed,
                'tasks_pending': checkpoint.tasks_pending,
                'last_event_seq': checkpoint.last_event_seq,
            }
        return None

    def reset_poison_detector(self) -> None:
        """Reset poison pill detector for new batch."""
        if self.poison_detector:
            self.poison_detector.reset()


async def execute_workflow_task(
    workflow_id: str,
    task_id: str,
    tool_registry: Optional[Dict[str, Callable[..., Awaitable[str]]]] = None,
    capture: Optional["DecisionCapture"] = None,
) -> ExecutionResult:
    """
    Convenience function to execute a single workflow task.

    Args:
        workflow_id: Workflow ID
        task_id: Task ID to execute
        tool_registry: Tool function registry
        capture: Optional decision capture

    Returns:
        ExecutionResult
    """
    executor = TaskExecutor(tool_registry=tool_registry, capture=capture)
    return await executor.execute_task(workflow_id, task_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo - would need actual tool registry in practice
    async def demo():
        _ = TaskExecutor()
        print(f"Agent to tool mapping: {len(AGENT_TOOL_MAPPING)} mappings")
        for agent, tool in AGENT_TOOL_MAPPING.items():
            print(f"  {agent} -> {tool}")

    asyncio.run(demo())
