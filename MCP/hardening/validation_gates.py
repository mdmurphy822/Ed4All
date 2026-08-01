"""
Validation Gate Framework

Implements fail-closed validation gates for phase transitions.
Supports configurable severity, thresholds, and waiver capture.

Phase 0 Hardening - Requirement 3: Hard Validation Gates
"""

import importlib
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

# W2.1: shared CUDA-OOM detection + free-VRAM probe. A CUDA OOM raised inside a
# validator was previously caught by the broad ``except`` in ``run_gate`` and,
# under ``behavior_on_error=warn``, rewritten to ``passed=True`` — a SILENT
# auto-pass that hid a broken (never-ran) gate. These helpers let the gate
# runner recognise the OOM and emit a DISTINCT ``VALIDATOR_OOM`` issue instead.
# Imported from ``lib.llm.oom`` (not ``MCP.core.executor``) to avoid a circular
# import — executor imports this module. Guarded so a stripped ``lib`` never
# breaks gate import; a failed import degrades OOM detection to off (the
# pre-W2.1 behaviour), never a crash.
try:
    from lib.llm.oom import is_cuda_oom, probe_free_vram_mib
except Exception:  # pragma: no cover - lib.llm always present in this repo
    def is_cuda_oom(exc: Optional[BaseException]) -> bool:  # type: ignore
        return False

    def probe_free_vram_mib() -> Optional[int]:  # type: ignore
        return None


# W2.1 opt-in escalation. Default OFF → a validator CUDA-OOM surfaces a
# DISTINCT warning-severity ``VALIDATOR_OOM`` issue but still honours the
# gate's configured ``behavior_on_error`` for pass/block (so it is no longer a
# SILENT pass, but it does not newly block a run that used to pass). ON → the
# OOM fails the gate closed (blocks) regardless of ``behavior_on_error``.
# Parse-with-fallback: only the explicit truthy tokens enable it.
_VALIDATOR_FAIL_CLOSED_ON_OOM_ENV = "ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM"


def _validator_fail_closed_on_oom() -> bool:
    """Return True iff ``ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM`` is truthy.

    Read at call time (not import) so tests can toggle it per-run. Accepts
    ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive); anything else —
    including unset / garbage — is off.
    """
    raw = os.environ.get(_VALIDATOR_FAIL_CLOSED_ON_OOM_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# Typed embedding-backend errors, same bug class as the CUDA-OOM above: a
# validator that raised one landed in the broad ``except`` below and, under
# ``behavior_on_error=warn``, was rewritten to ``passed=True`` — the exact
# silent auto-pass ``lib/llm/oom.py`` records for the OOM path. The two
# embedding errors are DISTINCT types with DIFFERENT contracts and must stay
# that way (neither subclasses the other — asserted in
# ``lib/embedding/tests/test_sentence_embedder.py``):
#
# * ``EmbeddingModelUnavailable`` — the ``[embedding]`` extras ARE installed
#   and the requested ``ED4ALL_EMBEDDING_DEVICE`` (default ``cuda``) did not
#   come up. FATAL, always: no env escape hatch, no ``behavior_on_error``
#   honouring. The owner rule is that a CUDA-unavailable embedding backend
#   never degrades silently; the operator opts out explicitly by pinning
#   ``ED4ALL_EMBEDDING_DEVICE=cpu``, which is the knob the failure message
#   names. (Contrast the OOM escalation, which IS env-gated: an OOM is a
#   transient resource condition, a missing device is a configuration fact.)
# * ``EmbeddingDepsMissing`` — the optional-extras contract. Only ever raised
#   when ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is on (default mode returns ``None``
#   from ``try_load_embedder`` and the validator degrades to a warning-severity
#   ``EMBEDDING_DEPS_MISSING`` with ``passed=True``, never reaching this
#   module's ``except``). So it fails closed under strict mode regardless of
#   ``behavior_on_error``, and otherwise keeps honouring ``on_error: warn`` —
#   the extras contract is UNCHANGED by the device passthrough.
#
# Guarded import so a stripped ``lib`` never breaks gate import; a failed
# import degrades typed detection to off (pre-fix behaviour), never a crash.
try:
    from lib.embedding.sentence_embedder import (
        EmbeddingDepsMissing as _EmbeddingDepsMissing,
        EmbeddingModelUnavailable as _EmbeddingModelUnavailable,
        is_strict_mode as _embedding_strict_mode,
    )
except Exception:  # pragma: no cover - lib.embedding always present in this repo
    _EmbeddingDepsMissing = None  # type: ignore[assignment]
    _EmbeddingModelUnavailable = None  # type: ignore[assignment]

    def _embedding_strict_mode() -> bool:  # type: ignore[misc]
        return False


#: Operator knob named in every device-unavailability message/suggestion.
_EMBEDDING_DEVICE_ENV = "ED4ALL_EMBEDDING_DEVICE"

#: Depth cap for the ``__cause__`` / ``__context__`` walk below.
_EXC_CHAIN_MAX_DEPTH = 10


def _exc_chain_has(exc: Optional[BaseException], exc_type: Any) -> bool:
    """True iff ``exc`` — or anything it was raised ``from`` — is ``exc_type``.

    Walks ``__cause__`` then ``__context__`` so a validator that re-raises the
    typed error wrapped (``raise RuntimeError(...) from exc``) is still
    recognised; a plain ``isinstance`` would miss it and the gate would fall
    back to the silent-auto-pass path this passthrough exists to close.
    Bounded depth + a seen-set so a self-referential chain cannot spin. Never
    raises; ``exc_type is None`` (guarded import failed) is always False.
    """
    if exc is None or exc_type is None:
        return False
    seen: set = set()
    cur: Optional[BaseException] = exc
    for _ in range(_EXC_CHAIN_MAX_DEPTH):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        try:
            if isinstance(cur, exc_type):
                return True
        except TypeError:  # pragma: no cover - defensive
            return False
        cur = cur.__cause__ or cur.__context__
    return False


class GateSeverity(Enum):
    """Gate severity levels."""
    CRITICAL = "critical"    # Blocks progression
    WARNING = "warning"      # Logged but doesn't block
    INFO = "info"            # Informational only


class GateBehavior(Enum):
    """Gate behavior on failure or error."""
    BLOCK = "block"              # Stop phase progression
    WARN = "warn"                # Log warning, continue
    FAIL_CLOSED = "fail_closed"  # Block on any error (safest)


@dataclass
class GateIssue:
    """Single validation issue found by a gate."""
    severity: str  # "critical", "warning", "info"
    code: str      # Machine-readable code
    message: str   # Human-readable message
    location: Optional[str] = None     # File/line/element location
    suggestion: Optional[str] = None   # How to fix

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GateResult:
    """Result from a validation gate."""
    gate_id: str
    validator_name: str
    validator_version: str
    passed: bool
    score: Optional[float] = None
    issues: List[GateIssue] = field(default_factory=list)
    execution_time_ms: int = 0
    inputs_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    waived: bool = False
    waiver_info: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    # Phase 3 Subtask 46: validator action signal consumed by the
    # Courseforge two-pass router. Legacy validators don't set this;
    # the router calls `derive_default_action()` to interpret them as
    # "pass" on success / "block" on failure.
    action: Optional[Literal["pass", "regenerate", "escalate", "block"]] = None
    # Wave W-D11 T11.1: optional per-validator aggregate counters /
    # diagnostic signals surfaced for downstream aggregators (e.g.
    # T11.5 promotion-chain rollup) without polluting the issues[]
    # surface. Validators that don't set this leave it ``None``;
    # consumers MUST handle ``metadata is None`` as "no signal" rather
    # than "zero". Free-form dict shape is intentional — each validator
    # owns its own keys, and downstream aggregators read by validator
    # name + key.
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['issues'] = [i if isinstance(i, dict) else i.to_dict() for i in self.issues]
        return d

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        """Count of warning issues."""
        return sum(1 for i in self.issues if i.severity == "warning")

    @classmethod
    def derive_default_action(cls, passed: bool, action: Optional[str]) -> str:
        """Resolve a router-consumable action for a (passed, action) pair.

        New Phase-3/4 validators set `action` directly. Legacy validators
        leave it `None`; treat them as "pass" on success / "block" on
        failure for back-compat.
        """
        if action is not None:
            return action
        return "pass" if passed else "block"


@dataclass
class GateWaiver:
    """Waiver for a failed gate."""
    gate_id: str
    who: str
    reason: str  # Must be 20+ chars for audit trail
    remediation_plan: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def validate(self) -> List[str]:
        """Validate waiver requirements."""
        issues = []
        if len(self.reason) < 20:
            issues.append("Waiver reason must be at least 20 characters")
        if not self.who:
            issues.append("Waiver must specify who approved it")
        if not self.remediation_plan:
            issues.append("Waiver must include remediation plan")
        return issues


class Validator(Protocol):
    """Protocol for validation gate implementations."""
    name: str
    version: str

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Run validation and return result."""
        ...


@dataclass
class GateConfig:
    """Configuration for a validation gate."""
    gate_id: str
    validator_path: str  # e.g., "lib.validators.wcag.WCAGValidator"
    severity: GateSeverity = GateSeverity.CRITICAL
    threshold: Dict[str, Any] = field(default_factory=dict)
    # Wave 78: arbitrary YAML ``config:`` block forwarded into the
    # validator's input dict (under ``_gate_config`` and merged at the
    # top level). Validators ignore unknown keys; opt-in flags like
    # ``strict``, ``strict_coverage``, ``strict_typing`` for the LibV2
    # packet integrity validator are read from this block.
    config: Dict[str, Any] = field(default_factory=dict)
    behavior_on_fail: GateBehavior = GateBehavior.BLOCK
    behavior_on_error: GateBehavior = GateBehavior.FAIL_CLOSED
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict) -> "GateConfig":
        """Create from dictionary (e.g., from YAML config)."""
        # Handle nested behavior dict
        behavior = data.pop('behavior', {})
        on_fail = behavior.get('on_fail', 'block')
        on_error = behavior.get('on_error', 'fail_closed')

        return cls(
            gate_id=data.get('gate_id', ''),
            validator_path=data.get('validator', ''),
            severity=GateSeverity(data.get('severity', 'critical')),
            threshold=data.get('threshold', {}),
            config=data.get('config', {}) or {},
            behavior_on_fail=GateBehavior(on_fail),
            behavior_on_error=GateBehavior(on_error),
            enabled=data.get('enabled', True)
        )


class ValidationGateManager:
    """Manages validation gates for workflow phases."""

    def __init__(self, capture: Any = None):
        """Initialize gate manager.

        Args:
            capture: Optional ``DecisionCapture`` instance threaded by
                the orchestrator's ``WorkflowExecutor`` (H3 Worker S0.5).
                Mirrored into ``inputs`` under both ``decision_capture``
                (Pattern A) and ``capture`` (Pattern B) keys at
                ``run_gate`` time so emitting validators see a live
                capture on direct invocations (tests / future MCP
                surfaces). ``None`` keeps the pre-S0.5 contract.
        """
        self._validators: Dict[str, Validator] = {}
        self._waivers: Dict[str, GateWaiver] = {}
        self._results_history: List[GateResult] = []
        self._capture = capture

    # Allowlist of module prefixes permitted for validator imports.
    # Prevents arbitrary module loading (e.g., os, subprocess) via config.
    ALLOWED_VALIDATOR_PREFIXES = (
        "lib.validators.",
        "lib.leak_checker",
        "Courseforge.router.",
    )

    def load_validator(self, validator_path: str) -> Validator:
        """
        Dynamically load a validator class.

        Args:
            validator_path: Full path to validator class
                           (e.g., "lib.validators.wcag.WCAGValidator")

        Returns:
            Validator instance

        Raises:
            ImportError: If module not found or not in allowlist
            AttributeError: If class not found in module
        """
        if validator_path in self._validators:
            return self._validators[validator_path]

        module_path, class_name = validator_path.rsplit('.', 1)

        # Security: only allow imports from known validator modules
        if not any(module_path.startswith(p) for p in self.ALLOWED_VALIDATOR_PREFIXES):
            raise ImportError(
                f"Validator module '{module_path}' not in allowlist. "
                f"Allowed prefixes: {self.ALLOWED_VALIDATOR_PREFIXES}"
            )

        module = importlib.import_module(module_path)
        validator_class = getattr(module, class_name)
        validator = validator_class()
        self._validators[validator_path] = validator

        logger.debug(f"Loaded validator: {validator_path}")
        return validator

    def run_gate(
        self,
        gate_config: GateConfig,
        inputs: Dict[str, Any]
    ) -> GateResult:
        """
        Run a single validation gate.

        Args:
            gate_config: Gate configuration
            inputs: Input data for validation

        Returns:
            GateResult with pass/fail and any issues
        """
        if not gate_config.enabled:
            return GateResult(
                gate_id=gate_config.gate_id,
                validator_name=gate_config.validator_path,
                validator_version="disabled",
                passed=True
            )

        start_time = datetime.now()

        try:
            validator = self.load_validator(gate_config.validator_path)
            # Wave 78: merge gate-config block into inputs so validators
            # can read opt-in flags (e.g., ``strict`` for packet
            # integrity) without per-builder plumbing. Existing
            # validators ignore unknown keys.
            if gate_config.config:
                merged_inputs: Dict[str, Any] = dict(inputs or {})
                for k, v in gate_config.config.items():
                    merged_inputs.setdefault(k, v)
                merged_inputs["_gate_config"] = dict(gate_config.config)
                inputs = merged_inputs
            # Forward the gate's declared ``threshold:`` dial into the
            # validator inputs. ``_apply_thresholds`` (below) only knows the
            # result-level keys (``max_critical_issues`` / ``max_issues`` /
            # ``min_score`` / ``required_score``); validators that read
            # per-dimension floors from their inputs (e.g. KGQualityValidator's
            # ``min_completeness`` / ``min_consistency`` / ``min_accuracy`` /
            # ``min_coverage``; MinEdgeCountValidator's ``min_edges`` etc.;
            # CurieAnchoringValidator's ``min_pair_anchoring_rate``) otherwise
            # never see the YAML-configured floor and silently fall back to
            # their built-in default (0.0 for kg_quality, which disables the
            # floor entirely). ``setdefault`` so a per-builder / per-call
            # override still wins; result-level keys consumed only by
            # ``_apply_thresholds`` are harmless extra inputs validators ignore.
            if gate_config.threshold:
                merged_thresholds: Dict[str, Any] = dict(inputs or {})
                for k, v in gate_config.threshold.items():
                    merged_thresholds.setdefault(k, v)
                inputs = merged_thresholds
            # H3 Worker S0.5: mirror the executor-side capture injection
            # so direct callers (test harness / future MCP-exposed
            # validate tools that build inputs by hand) also see a live
            # capture. Idempotent with the executor seam — the executor
            # injects first, this ``setdefault`` is a no-op when both
            # fire. ``setdefault`` so explicit per-call overrides win.
            if self._capture is not None:
                merged_inputs2: Dict[str, Any] = dict(inputs or {})
                merged_inputs2.setdefault("decision_capture", self._capture)
                merged_inputs2.setdefault("capture", self._capture)
                inputs = merged_inputs2
            result = validator.validate(inputs)
            result.gate_id = gate_config.gate_id

            # Apply threshold checks
            result = self._apply_thresholds(result, gate_config.threshold)

        except Exception as e:
            # W2.1: a CUDA out-of-memory raised INSIDE a validator (NLI /
            # embedding forward pass on a VRAM-starved box) was previously
            # indistinguishable from any other validator bug here — and under
            # ``behavior_on_error=warn`` it became a SILENT auto-pass. Detect
            # the OOM and emit a DISTINCT, greppable ``VALIDATOR_OOM`` issue
            # (plus a DecisionCapture) so the OOM is never invisible; honour
            # the opt-in ``ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM`` to block.
            #
            # Same bug class, different typed errors: the two embedding
            # backend errors are checked FIRST. Order matters against the OOM
            # branch — ``SentenceEmbedder._ensure_model`` wraps ANY construction
            # failure (a CUDA OOM included) in ``EmbeddingModelUnavailable``, so
            # a message-sniffing ``is_cuda_oom`` would also match it and, with
            # the OOM escalation flag off, hand it back to
            # ``behavior_on_error=warn`` — i.e. the silent pass again. The
            # device branch is unconditionally fatal, so checking it first is
            # the strictly safer resolution of that overlap.
            if _exc_chain_has(e, _EmbeddingModelUnavailable):
                result = self._build_embedding_device_gate_result(gate_config, e)
            elif _exc_chain_has(e, _EmbeddingDepsMissing):
                result = self._build_embedding_deps_missing_gate_result(gate_config, e)
            elif is_cuda_oom(e):
                result = self._build_oom_gate_result(gate_config, e)
            else:
                logger.error(f"Validator error for gate {gate_config.gate_id}: {e}")

                # Fail-closed on error by default
                result = GateResult(
                    gate_id=gate_config.gate_id,
                    validator_name=gate_config.validator_path,
                    validator_version="error",
                    passed=False,
                    error=str(e),
                    issues=[GateIssue(
                        severity="critical",
                        code="VALIDATOR_ERROR",
                        message=f"Validator threw exception: {e}"
                    )]
                )

                # Check behavior on error
                if gate_config.behavior_on_error == GateBehavior.WARN:
                    result.passed = True
                    logger.warning(f"Gate {gate_config.gate_id} error treated as warning")

        end_time = datetime.now()
        result.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Check for waiver
        if gate_config.gate_id in self._waivers:
            waiver = self._waivers[gate_config.gate_id]

            # Check if waiver is expired
            if waiver.expires_at:
                expires = datetime.fromisoformat(waiver.expires_at)
                if datetime.now() > expires:
                    logger.info(f"Waiver for gate {gate_config.gate_id} has expired")
                else:
                    result.waived = True
                    result.waiver_info = waiver.to_dict()
                    result.passed = True
                    logger.info(f"Gate {gate_config.gate_id} passed via waiver")
            else:
                result.waived = True
                result.waiver_info = waiver.to_dict()
                result.passed = True
                logger.info(f"Gate {gate_config.gate_id} passed via waiver")

        # Store result
        self._results_history.append(result)

        return result

    def _build_oom_gate_result(
        self,
        gate_config: GateConfig,
        exc: BaseException,
    ) -> GateResult:
        """Build the ``VALIDATOR_OOM`` gate result for a validator CUDA-OOM.

        W2.1: replaces the pre-fix behaviour where a CUDA OOM inside a
        validator was caught by the generic ``except`` and, under
        ``behavior_on_error=warn``, silently auto-passed. Pass/block is
        resolved so:

        * ``ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM`` ON → always fail closed
          (``passed=False``, critical issue) regardless of the gate's
          ``behavior_on_error``.
        * flag OFF → honour ``behavior_on_error`` for pass/block (WARN →
          ``passed=True``; BLOCK / FAIL_CLOSED → ``passed=False``), but ALWAYS
          emit a DISTINCT ``VALIDATOR_OOM`` issue + a DecisionCapture, so even
          the non-blocking case is greppable and never a silent pass.
        """
        fail_closed = _validator_fail_closed_on_oom()
        if fail_closed:
            passed = False
        else:
            passed = gate_config.behavior_on_error == GateBehavior.WARN

        free_mib = probe_free_vram_mib()
        free_desc = (
            f"{free_mib} MiB free"
            if free_mib is not None
            else "free VRAM unprobeable"
        )
        severity = "warning" if passed else "critical"

        logger.error(
            f"GPU OUT OF MEMORY during validation gate {gate_config.gate_id} "
            f"({gate_config.validator_path}): {free_desc} at failure — a "
            f"resident model (likely the local 7B) is starving NLI/embedding "
            f"scoring. "
            + (
                "Failing the gate closed (blocking)"
                if not passed
                else "Emitting a VALIDATOR_OOM warning (non-blocking, NOT a "
                "silent pass)"
            )
            + (
                " [ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM=on]" if fail_closed else ""
            )
            + f". Exception: {exc}"
        )

        result = GateResult(
            gate_id=gate_config.gate_id,
            validator_name=gate_config.validator_path,
            validator_version="oom",
            passed=passed,
            error=str(exc),
            issues=[GateIssue(
                severity=severity,
                code="VALIDATOR_OOM",
                message=(
                    f"Validator hit CUDA out-of-memory ({free_desc} at "
                    f"failure): {exc}. The gate did NOT run to completion — "
                    f"this is a GPU OOM, not a pass."
                ),
                suggestion=(
                    "Free VRAM before re-running (evict the resident local "
                    "LLM, raise the free-VRAM floor via "
                    "ED4ALL_NLI_MIN_FREE_VRAM_MIB, or pin NLI/embeddings to "
                    "CPU). Set ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM=1 to block "
                    "the phase on any validator OOM."
                ),
            )],
        )

        # DecisionCapture on the OOM branch (dynamic, replayable rationale).
        if self._capture is not None:
            try:
                self._capture.log_decision(
                    decision_type="validation_result",
                    decision=(
                        f"Gate {gate_config.gate_id} hit CUDA OOM; emitted a "
                        f"distinct VALIDATOR_OOM issue "
                        + (
                            "(fail-closed/block)"
                            if not passed
                            else "(warning/non-blocking)"
                        )
                    ),
                    rationale=(
                        f"Validator {gate_config.validator_path} raised a CUDA "
                        f"out-of-memory ({free_desc} at failure); surfacing a "
                        f"distinct VALIDATOR_OOM issue instead of the pre-W2.1 "
                        f"silent auto-pass. "
                        f"ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM="
                        f"{'on' if fail_closed else 'off'}, "
                        f"behavior_on_error={gate_config.behavior_on_error.value}, "
                        f"resolved passed={passed}"
                    ),
                )
            except Exception:  # noqa: BLE001 - capture must never break a gate
                pass

        return result

    def _build_embedding_device_gate_result(
        self,
        gate_config: GateConfig,
        exc: BaseException,
    ) -> GateResult:
        """Build the ``EMBEDDING_MODEL_UNAVAILABLE`` result — ALWAYS fail closed.

        ``EmbeddingModelUnavailable`` means the ``[embedding]`` extras are
        installed and the requested embedding DEVICE (``ED4ALL_EMBEDDING_DEVICE``,
        default ``cuda``) did not come up. The statistical-tier validators
        deliberately let that type propagate instead of swallowing it into a
        vacuous pass; this method is the other half of that fix — without it the
        generic handler below rewrites the raise back to ``passed=True`` for the
        twelve gate wirings configured ``on_error: warn``, and the
        validator-level fail-closed never reaches the pipeline.

        Unlike the OOM branch there is NO env escape hatch and
        ``behavior_on_error`` is NOT consulted: a CUDA-unavailable embedding
        backend is fatal by owner rule, and the documented opt-out is the
        explicit, greppable ``ED4ALL_EMBEDDING_DEVICE=cpu`` — not a permissive
        default. This does NOT touch the missing-extras contract, which lives on
        a distinct type (see ``_build_embedding_deps_missing_gate_result``).
        """
        logger.error(
            f"EMBEDDING BACKEND UNAVAILABLE during validation gate "
            f"{gate_config.gate_id} ({gate_config.validator_path}): the "
            f"[embedding] extras are installed but the requested "
            f"{_EMBEDDING_DEVICE_ENV} device did not come up, so the gate never "
            f"ran. Failing the gate closed regardless of behavior_on_error="
            f"{gate_config.behavior_on_error.value} (a device-unavailable "
            f"embedding backend is never a degrade). Set "
            f"{_EMBEDDING_DEVICE_ENV}=cpu to run this tier on CPU, or provision "
            f"the requested device. Exception: {exc}"
        )

        result = GateResult(
            gate_id=gate_config.gate_id,
            validator_name=gate_config.validator_path,
            validator_version="embedding_device_unavailable",
            passed=False,
            error=str(exc),
            issues=[GateIssue(
                severity="critical",
                code="EMBEDDING_MODEL_UNAVAILABLE",
                message=(
                    f"Embedding model could not be constructed on the requested "
                    f"{_EMBEDDING_DEVICE_ENV} device: {exc}. The gate did NOT "
                    f"run to completion — this is an unavailable embedding "
                    f"backend, not a pass, and it is NOT the missing-extras "
                    f"degrade (those extras are installed)."
                ),
                suggestion=(
                    f"Provision the requested device, or opt out explicitly "
                    f"with {_EMBEDDING_DEVICE_ENV}=cpu (there is no automatic "
                    f"CUDA→CPU downgrade and no behavior_on_error override — "
                    f"the CPU choice must be made by the operator)."
                ),
            )],
        )

        # DecisionCapture on the device branch (dynamic, replayable rationale).
        if self._capture is not None:
            try:
                self._capture.log_decision(
                    decision_type="validation_result",
                    decision=(
                        f"Gate {gate_config.gate_id} failed closed on an "
                        f"unavailable embedding device (EMBEDDING_MODEL_UNAVAILABLE)"
                    ),
                    rationale=(
                        f"Validator {gate_config.validator_path} raised "
                        f"EmbeddingModelUnavailable — the [embedding] extras are "
                        f"present but the requested {_EMBEDDING_DEVICE_ENV} device "
                        f"did not come up, so the gate never adjudicated anything. "
                        f"Failing closed despite behavior_on_error="
                        f"{gate_config.behavior_on_error.value}; the documented "
                        f"opt-out is {_EMBEDDING_DEVICE_ENV}=cpu. Exception: {exc}"
                    ),
                )
            except Exception:  # noqa: BLE001 - capture must never break a gate
                pass

        return result

    def _build_embedding_deps_missing_gate_result(
        self,
        gate_config: GateConfig,
        exc: BaseException,
    ) -> GateResult:
        """Build the ``EMBEDDING_DEPS_MISSING`` result for a RAISED deps error.

        This is the optional-extras contract, and it is deliberately NOT changed
        by the device passthrough above:

        * ``TRAINFORGE_REQUIRE_EMBEDDINGS`` off — honour ``behavior_on_error``
          exactly as before (``warn`` → ``passed=True``), but with a distinct
          warning-severity ``EMBEDDING_DEPS_MISSING`` code instead of the generic
          ``VALIDATOR_ERROR``, so a degrade stays distinguishable from a real
          validator bug in downstream rollups. (In practice the default path
          never even raises: ``try_load_embedder`` returns ``None`` and the
          validator emits this same code itself with ``passed=True``.)
        * strict mode on — ``try_load_embedder`` RAISES, and strict mode means
          fail closed. Honouring ``on_error: warn`` there would silently undo
          the operator's own opt-in, so strict mode blocks regardless of
          ``behavior_on_error``.
        """
        strict = False
        try:
            strict = bool(_embedding_strict_mode())
        except Exception:  # noqa: BLE001 - resolution must never break a gate
            strict = False

        passed = (not strict) and gate_config.behavior_on_error == GateBehavior.WARN
        severity = "warning" if passed else "critical"

        log = logger.error if not passed else logger.warning
        log(
            f"Embedding extras unavailable during validation gate "
            f"{gate_config.gate_id} ({gate_config.validator_path}): "
            + (
                "TRAINFORGE_REQUIRE_EMBEDDINGS is on, so the gate fails closed "
                "regardless of behavior_on_error"
                if strict
                else f"honouring behavior_on_error="
                f"{gate_config.behavior_on_error.value} (install the "
                f"[embedding] extras to enable this tier)"
            )
            + f". Exception: {exc}"
        )

        result = GateResult(
            gate_id=gate_config.gate_id,
            validator_name=gate_config.validator_path,
            validator_version="embedding_deps_missing",
            passed=passed,
            error=str(exc),
            issues=[GateIssue(
                severity=severity,
                code="EMBEDDING_DEPS_MISSING",
                message=(
                    f"Embedding extras are not installed, so this gate could not "
                    f"run: {exc}. This is the optional-extras degrade, NOT an "
                    f"unavailable embedding device."
                ),
                suggestion=(
                    "Install the extras via `pip install -e .[embedding]`. "
                    "TRAINFORGE_REQUIRE_EMBEDDINGS=true makes a missing-extras "
                    "degrade fail the gate closed."
                ),
            )],
        )

        if self._capture is not None:
            try:
                self._capture.log_decision(
                    decision_type="validation_result",
                    decision=(
                        f"Gate {gate_config.gate_id} hit missing [embedding] "
                        f"extras; emitted EMBEDDING_DEPS_MISSING "
                        + ("(fail-closed/block)" if not passed else "(warning/non-blocking)")
                    ),
                    rationale=(
                        f"Validator {gate_config.validator_path} raised "
                        f"EmbeddingDepsMissing — the optional-extras contract, "
                        f"distinct from an unavailable embedding device. "
                        f"TRAINFORGE_REQUIRE_EMBEDDINGS={'on' if strict else 'off'}, "
                        f"behavior_on_error={gate_config.behavior_on_error.value}, "
                        f"resolved passed={passed}"
                    ),
                )
            except Exception:  # noqa: BLE001 - capture must never break a gate
                pass

        return result

    def _apply_thresholds(
        self,
        result: GateResult,
        threshold: Dict[str, Any]
    ) -> GateResult:
        """Apply threshold checks to gate result."""
        if not threshold:
            return result

        # Check max critical issues
        if 'max_critical_issues' in threshold:
            max_critical = threshold['max_critical_issues']
            if result.critical_count > max_critical:
                result.passed = False
                logger.debug(
                    f"Gate failed: {result.critical_count} critical issues "
                    f"> {max_critical} threshold"
                )

        # Check max total issues
        if 'max_issues' in threshold:
            max_issues = threshold['max_issues']
            if len(result.issues) > max_issues:
                result.passed = False

        # Check minimum score
        if 'min_score' in threshold:
            min_score = threshold['min_score']
            if result.score is not None and result.score < min_score:
                result.passed = False
                logger.debug(
                    f"Gate failed: score {result.score} < {min_score} threshold"
                )

        # Check required score
        if 'required_score' in threshold:
            required = threshold['required_score']
            if result.score is None or result.score < required:
                result.passed = False

        return result

    def run_phase_gates(
        self,
        phase_name: str,
        gate_configs: List[GateConfig],
        inputs: Dict[str, Any]
    ) -> Tuple[bool, List[GateResult]]:
        """
        Run all gates for a phase.

        Args:
            phase_name: Name of the phase
            gate_configs: List of gate configurations
            inputs: Input data for validation

        Returns:
            Tuple of (all_passed, list of results)
        """
        results = []
        all_passed = True

        logger.info(f"Running {len(gate_configs)} validation gates for phase: {phase_name}")

        for gate_config in gate_configs:
            if not gate_config.enabled:
                logger.debug(f"Skipping disabled gate: {gate_config.gate_id}")
                continue

            result = self.run_gate(gate_config, inputs)
            results.append(result)

            if not result.passed:
                if gate_config.severity == GateSeverity.CRITICAL:
                    all_passed = False
                    logger.warning(
                        f"Critical gate failed: {gate_config.gate_id} "
                        f"({result.critical_count} critical, {result.warning_count} warnings)"
                    )

                    if gate_config.behavior_on_fail == GateBehavior.BLOCK:
                        logger.info("Stopping gate evaluation due to blocking failure")
                        break
                elif gate_config.severity == GateSeverity.WARNING:
                    logger.warning(
                        f"Warning gate failed (non-blocking): {gate_config.gate_id}"
                    )

        return all_passed, results

    def add_waiver(self, waiver: GateWaiver) -> List[str]:
        """
        Add a waiver for a gate.

        Args:
            waiver: The waiver to add

        Returns:
            List of validation issues (empty if valid)
        """
        issues = waiver.validate()
        if issues:
            logger.warning(f"Invalid waiver for gate {waiver.gate_id}: {issues}")
            return issues

        self._waivers[waiver.gate_id] = waiver
        logger.info(
            f"Added waiver for gate {waiver.gate_id} "
            f"by {waiver.who}: {waiver.reason[:50]}..."
        )
        return []

    def get_waiver(self, gate_id: str) -> Optional[GateWaiver]:
        """Get waiver for a gate if exists."""
        return self._waivers.get(gate_id)

    def remove_waiver(self, gate_id: str) -> bool:
        """Remove a waiver."""
        if gate_id in self._waivers:
            del self._waivers[gate_id]
            logger.info(f"Removed waiver for gate {gate_id}")
            return True
        return False

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all gate results."""
        passed = [r for r in self._results_history if r.passed]
        failed = [r for r in self._results_history if not r.passed]
        waived = [r for r in self._results_history if r.waived]

        return {
            "total_gates": len(self._results_history),
            "passed": len(passed),
            "failed": len(failed),
            "waived": len(waived),
            "total_critical_issues": sum(r.critical_count for r in self._results_history),
            "total_warnings": sum(r.warning_count for r in self._results_history),
            "avg_execution_time_ms": (
                sum(r.execution_time_ms for r in self._results_history) / len(self._results_history)
                if self._results_history else 0
            )
        }


# Built-in validators

class SchemaValidator:
    """Built-in JSON Schema validator."""
    name = "schema_validator"
    version = "1.0.0"

    def __init__(self, schema: Optional[Dict] = None):
        """Initialize with optional schema."""
        self.schema = schema

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate inputs against JSON schema."""
        issues = []

        schema = inputs.get('schema') or self.schema
        data = inputs.get('data')

        if not schema:
            return GateResult(
                gate_id="schema_validation",
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                error="No schema provided"
            )

        try:
            import jsonschema
            jsonschema.validate(data, schema)
            passed = True
        except jsonschema.ValidationError as e:
            passed = False
            issues.append(GateIssue(
                severity="critical",
                code="SCHEMA_VALIDATION_ERROR",
                message=str(e.message),
                location=".".join(str(p) for p in e.absolute_path)
            ))
        except ImportError:
            # jsonschema not installed
            passed = True
            issues.append(GateIssue(
                severity="warning",
                code="JSONSCHEMA_NOT_INSTALLED",
                message="jsonschema library not installed, skipping validation"
            ))

        return GateResult(
            gate_id="schema_validation",
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            issues=issues
        )


class FileExistsValidator:
    """Built-in validator that checks required files exist."""
    name = "file_exists_validator"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate that required files exist."""
        from pathlib import Path

        issues = []
        required_files = inputs.get('required_files', [])

        for file_path in required_files:
            path = Path(file_path)
            if not path.exists():
                issues.append(GateIssue(
                    severity="critical",
                    code="MISSING_FILE",
                    message=f"Required file not found: {file_path}",
                    location=str(file_path)
                ))

        return GateResult(
            gate_id="file_exists",
            validator_name=self.name,
            validator_version=self.version,
            passed=len(issues) == 0,
            issues=issues
        )


# Convenience functions

def create_gate_from_config(config_dict: Dict) -> GateConfig:
    """Create GateConfig from dictionary."""
    return GateConfig.from_dict(config_dict)


def run_validation_gates(
    gate_configs: List[Dict],
    inputs: Dict[str, Any]
) -> Tuple[bool, List[Dict]]:
    """
    Convenience function to run gates from config dictionaries.

    Args:
        gate_configs: List of gate config dictionaries
        inputs: Input data for validation

    Returns:
        Tuple of (all_passed, list of result dictionaries)
    """
    manager = ValidationGateManager()
    configs = [GateConfig.from_dict(c) for c in gate_configs]
    passed, results = manager.run_phase_gates("validation", configs, inputs)
    return passed, [r.to_dict() for r in results]
