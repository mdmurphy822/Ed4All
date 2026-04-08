"""
Audit Logger - Unified Audit Logging with Redaction

Provides unified audit logging across all Ed4All components:
- File access logging
- Tool invocation logging
- State change logging
- Workflow event logging
- Security event logging

All events share run_id and use hash-chaining for tamper evidence.

Phase 0 Hardening: Requirement 5 (Audit Logging Completeness)
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .hash_chain import HashChainedLog, GENESIS_HASH
from .provenance import hash_content, hash_file
from .run_manager import get_current_run, RUNS_PATH


# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================

class EventType(str, Enum):
    """Types of audit events."""
    FILE_ACCESS = "file_access"
    TOOL_INVOCATION = "tool_invocation"
    STATE_CHANGE = "state_change"
    DECISION_EVENT = "decision_event"
    WORKFLOW_EVENT = "workflow_event"
    VALIDATION_EVENT = "validation_event"
    ERROR = "error"
    SECURITY_EVENT = "security_event"
    PROMPT_INVOCATION = "prompt_invocation"  # Phase 0.5: Prompt fingerprinting


class FileOperation(str, Enum):
    """File operation types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    CREATE = "create"
    LIST = "list"


class SecurityEventType(str, Enum):
    """Security event subtypes."""
    SANDBOX_VIOLATION = "sandbox_violation"
    PERMISSION_DENIED = "permission_denied"
    SECRET_DETECTED = "secret_detected"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"


# Default redaction patterns (can be overridden by config)
DEFAULT_REDACTION_PATTERNS = [
    r'(?i)(api[_-]?key|secret|password|token|credential)["\']?\s*[:=]\s*["\']?[^"\'\s]+',
    r'(?i)bearer\s+[a-zA-Z0-9._-]+',
    r'sk-[a-zA-Z0-9]{48}',  # OpenAI keys
    r'claude-[a-zA-Z0-9-]+',  # Anthropic keys
    r'(?i)authorization:\s*\S+',
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FileAccessDetails:
    """Details for file access events."""
    operation: str
    path: str
    content_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ToolInvocationDetails:
    """Details for tool invocation events."""
    tool_name: str
    tool_version: str = ""
    args_hash: str = ""
    result_hash: str = ""
    duration_ms: float = 0
    exit_status: int = 0
    success: bool = True
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StateChangeDetails:
    """Details for state change events."""
    state_file: str
    change_type: str  # create, update, delete, lock, unlock
    previous_hash: Optional[str] = None
    new_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class WorkflowEventDetails:
    """Details for workflow events."""
    workflow_id: str
    event_subtype: str  # start, phase_start, phase_complete, task_dispatch, etc.
    phase: Optional[str] = None
    task_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if not result.get("task_ids"):
            del result["task_ids"]
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class ValidationEventDetails:
    """Details for validation events."""
    validator_name: str
    passed: bool
    validator_version: str = ""
    target: str = ""
    score: Optional[float] = None
    issue_count: int = 0
    waived: bool = False
    waiver_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ErrorDetails:
    """Details for error events."""
    error_type: str
    message: str
    error_class: str = ""
    recoverable: bool = False
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if not result.get("context"):
            del result["context"]
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class SecurityEventDetails:
    """Details for security events."""
    security_event_type: str
    attempted_action: str = ""
    blocked: bool = True
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PromptInvocationDetails:
    """
    Details for prompt invocation events.

    Phase 0.5: Prompt fingerprinting for reproducibility.
    """
    model: str
    prompt_hash: str  # Hash of prompt text for reproducibility
    prompt_length: int = 0
    system_prompt_hash: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_hash: Optional[str] = None  # Hash of response for verification
    response_length: int = 0
    latency_ms: float = 0
    tokens_used: Optional[int] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ============================================================================
# REDACTION
# ============================================================================

class Redactor:
    """Handles redaction of sensitive data from audit logs."""

    def __init__(self, patterns: Optional[List[str]] = None):
        """
        Initialize redactor.

        Args:
            patterns: Regex patterns to redact. Uses defaults if not provided.
        """
        self.patterns = [re.compile(p) for p in (patterns or DEFAULT_REDACTION_PATTERNS)]
        self.redacted_fields: List[str] = []

    def redact_string(self, value: str) -> str:
        """Redact sensitive patterns from a string."""
        result = value
        for pattern in self.patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def redact_dict(self, data: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """
        Recursively redact sensitive data from a dictionary.

        Args:
            data: Dictionary to redact
            path: Current path for tracking redacted fields

        Returns:
            Redacted dictionary
        """
        result = {}
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            # Check if key name itself suggests sensitive data
            if any(s in key.lower() for s in ["password", "secret", "token", "key", "credential"]):
                result[key] = "[REDACTED]"
                self.redacted_fields.append(current_path)
            elif isinstance(value, str):
                redacted = self.redact_string(value)
                if redacted != value:
                    self.redacted_fields.append(current_path)
                result[key] = redacted
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, current_path)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_dict(item, f"{current_path}[{i}]") if isinstance(item, dict)
                    else self.redact_string(item) if isinstance(item, str)
                    else item
                    for i, item in enumerate(value)
                ]
            else:
                result[key] = value

        return result

    def get_redacted_fields(self) -> List[str]:
        """Get list of fields that were redacted."""
        fields = self.redacted_fields.copy()
        self.redacted_fields = []  # Reset for next use
        return fields


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """
    Unified audit logger for Ed4All.

    Uses hash-chaining for tamper evidence and supports redaction.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        log_path: Optional[Path] = None,
        enable_redaction: bool = True,
        redaction_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize audit logger.

        Args:
            run_id: Run ID for this logger. Uses current run if not provided.
            log_path: Path to audit log file. Defaults to run's audit path.
            enable_redaction: Whether to redact sensitive data
            redaction_patterns: Custom redaction patterns
        """
        # Determine run context
        self._run_context = get_current_run()
        if run_id:
            self.run_id = run_id
        elif self._run_context:
            self.run_id = self._run_context.run_id
        else:
            self.run_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine log path
        if log_path:
            self.log_path = log_path
        elif self._run_context:
            self.log_path = self._run_context.audit_path / "audit.jsonl"
        else:
            self.log_path = RUNS_PATH / "orphan_audits" / f"audit_{self.run_id}.jsonl"

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize hash-chained log
        self._chain = HashChainedLog(self.log_path)

        # Redaction
        self.enable_redaction = enable_redaction
        self._redactor = Redactor(redaction_patterns) if enable_redaction else None

        # Component context
        self._component: Optional[str] = None
        self._worker_id: Optional[str] = None
        self._task_id: Optional[str] = None

    def set_context(
        self,
        component: Optional[str] = None,
        worker_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Set context for subsequent log entries."""
        if component is not None:
            self._component = component
        if worker_id is not None:
            self._worker_id = worker_id
        if task_id is not None:
            self._task_id = task_id

    def _build_event(
        self,
        event_type: EventType,
        details: Union[Dict[str, Any], FileAccessDetails, ToolInvocationDetails,
                       StateChangeDetails, WorkflowEventDetails, ValidationEventDetails,
                       ErrorDetails, SecurityEventDetails],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a complete audit event."""
        from .sequence_manager import get_sequence_for_context, generate_event_id

        # Get sequence and event ID
        try:
            seq, event_id = get_sequence_for_context(self.run_id if self._run_context else None)
        except Exception:
            seq = 0
            event_id = generate_event_id()

        # Convert details to dict if needed
        if hasattr(details, 'to_dict'):
            details_dict = details.to_dict()
        else:
            details_dict = details

        event = {
            "run_id": self.run_id,
            "event_id": event_id,
            "seq": seq,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value if isinstance(event_type, EventType) else event_type,
            "component": self._component,
            "worker_id": self._worker_id,
            "task_id": self._task_id,
            "details": details_dict,
        }

        if metadata:
            event["metadata"] = metadata

        # Apply redaction
        redacted_fields = []
        if self._redactor:
            event = self._redactor.redact_dict(event)
            redacted_fields = self._redactor.get_redacted_fields()

        if redacted_fields:
            event["redacted_fields"] = redacted_fields

        return event

    def log(
        self,
        event_type: EventType,
        details: Union[Dict[str, Any], FileAccessDetails, ToolInvocationDetails,
                       StateChangeDetails, WorkflowEventDetails, ValidationEventDetails,
                       ErrorDetails, SecurityEventDetails],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            details: Event details (dataclass or dict)
            metadata: Additional metadata

        Returns:
            Event ID
        """
        event = self._build_event(event_type, details, metadata)
        chained_event = self._chain.append(event)
        return event["event_id"]

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    def log_file_access(
        self,
        operation: FileOperation,
        path: Union[str, Path],
        compute_hash: bool = True,
        success: bool = True,
    ) -> str:
        """Log a file access event."""
        path = Path(path)
        content_hash = None
        size_bytes = None

        if compute_hash and path.exists() and operation in [FileOperation.READ, FileOperation.WRITE]:
            try:
                content_hash = hash_file(path)[:12]  # First 12 chars
                size_bytes = path.stat().st_size
            except Exception:
                pass

        details = FileAccessDetails(
            operation=operation.value if isinstance(operation, FileOperation) else operation,
            path=str(path),
            content_hash=content_hash,
            size_bytes=size_bytes,
            success=success,
        )
        return self.log(EventType.FILE_ACCESS, details)

    def log_tool_invocation(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        duration_ms: float = 0,
        exit_status: int = 0,
        success: bool = True,
        error_type: Optional[str] = None,
        tool_version: str = "",
    ) -> str:
        """Log a tool invocation event."""
        args_hash = ""
        result_hash = ""

        if args:
            args_hash = hash_content(json.dumps(args, sort_keys=True))[:12]
        if result is not None:
            result_hash = hash_content(json.dumps(result, sort_keys=True, default=str))[:12]

        details = ToolInvocationDetails(
            tool_name=tool_name,
            tool_version=tool_version,
            args_hash=args_hash,
            result_hash=result_hash,
            duration_ms=duration_ms,
            exit_status=exit_status,
            success=success,
            error_type=error_type,
        )
        return self.log(EventType.TOOL_INVOCATION, details)

    def log_state_change(
        self,
        state_file: Union[str, Path],
        change_type: str,
        previous_hash: Optional[str] = None,
        new_hash: Optional[str] = None,
    ) -> str:
        """Log a state change event."""
        details = StateChangeDetails(
            state_file=str(state_file),
            change_type=change_type,
            previous_hash=previous_hash,
            new_hash=new_hash,
        )
        return self.log(EventType.STATE_CHANGE, details)

    def log_workflow_event(
        self,
        workflow_id: str,
        event_subtype: str,
        phase: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
    ) -> str:
        """Log a workflow event."""
        details = WorkflowEventDetails(
            workflow_id=workflow_id,
            event_subtype=event_subtype,
            phase=phase,
            task_ids=task_ids or [],
        )
        return self.log(EventType.WORKFLOW_EVENT, details)

    def log_validation_event(
        self,
        validator_name: str,
        passed: bool,
        validator_version: str = "",
        target: str = "",
        score: Optional[float] = None,
        issue_count: int = 0,
        waived: bool = False,
        waiver_reason: Optional[str] = None,
    ) -> str:
        """Log a validation event."""
        details = ValidationEventDetails(
            validator_name=validator_name,
            passed=passed,
            validator_version=validator_version,
            target=target,
            score=score,
            issue_count=issue_count,
            waived=waived,
            waiver_reason=waiver_reason,
        )
        return self.log(EventType.VALIDATION_EVENT, details)

    def log_error(
        self,
        error_type: str,
        message: str,
        error_class: str = "",
        recoverable: bool = False,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an error event."""
        import traceback

        stack_trace = None
        if exception:
            stack_trace = traceback.format_exception(type(exception), exception, exception.__traceback__)
            stack_trace = "".join(stack_trace)
            if not error_class:
                error_class = type(exception).__name__

        details = ErrorDetails(
            error_type=error_type,
            message=message,
            error_class=error_class,
            recoverable=recoverable,
            stack_trace=stack_trace,
            context=context or {},
        )
        return self.log(EventType.ERROR, details)

    def log_security_event(
        self,
        security_event_type: SecurityEventType,
        attempted_action: str = "",
        blocked: bool = True,
        source: str = "",
    ) -> str:
        """Log a security event."""
        details = SecurityEventDetails(
            security_event_type=security_event_type.value if isinstance(security_event_type, SecurityEventType) else security_event_type,
            attempted_action=attempted_action,
            blocked=blocked,
            source=source,
        )
        return self.log(EventType.SECURITY_EVENT, details)

    def log_prompt_invocation(
        self,
        prompt_text: str,
        model: str,
        system_prompt: Optional[str] = None,
        response_text: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        latency_ms: float = 0,
        tokens_used: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> str:
        """
        Log a prompt invocation with hash for reproducibility.

        Phase 0.5: Prompt fingerprinting support.

        Args:
            prompt_text: The prompt text sent to the model
            model: Model identifier (e.g., "claude-3-opus")
            system_prompt: Optional system prompt
            response_text: Optional response text
            temperature: Temperature setting
            max_tokens: Max tokens setting
            latency_ms: Response latency in milliseconds
            tokens_used: Total tokens used
            success: Whether the call succeeded
            error: Error message if failed

        Returns:
            Event ID
        """
        # Compute prompt hash
        prompt_hash = hash_content(prompt_text)[:16]

        # Compute system prompt hash if provided
        system_prompt_hash = None
        if system_prompt:
            system_prompt_hash = hash_content(system_prompt)[:16]

        # Compute response hash if provided
        response_hash = None
        if response_text:
            response_hash = hash_content(response_text)[:16]

        details = PromptInvocationDetails(
            model=model,
            prompt_hash=prompt_hash,
            prompt_length=len(prompt_text),
            system_prompt_hash=system_prompt_hash,
            temperature=temperature,
            max_tokens=max_tokens,
            response_hash=response_hash,
            response_length=len(response_text) if response_text else 0,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            success=success,
            error=error,
        )
        return self.log(EventType.PROMPT_INVOCATION, details)

    def verify(self):
        """Verify the audit log integrity."""
        return self._chain.verify()


# ============================================================================
# GLOBAL LOGGER ACCESS
# ============================================================================

_audit_loggers: Dict[str, AuditLogger] = {}


def get_audit_logger(run_id: Optional[str] = None) -> AuditLogger:
    """
    Get or create an audit logger for a run.

    Args:
        run_id: Run ID, or None to use current run

    Returns:
        AuditLogger for the run
    """
    run_context = get_current_run()

    if run_id is None:
        if run_context:
            run_id = run_context.run_id
        else:
            run_id = "default"

    if run_id not in _audit_loggers:
        _audit_loggers[run_id] = AuditLogger(run_id)

    return _audit_loggers[run_id]


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "EventType",
    "FileOperation",
    "SecurityEventType",
    # Data classes
    "FileAccessDetails",
    "ToolInvocationDetails",
    "StateChangeDetails",
    "WorkflowEventDetails",
    "ValidationEventDetails",
    "ErrorDetails",
    "SecurityEventDetails",
    "PromptInvocationDetails",  # Phase 0.5
    # Redaction
    "Redactor",
    "DEFAULT_REDACTION_PATTERNS",
    # Logger
    "AuditLogger",
    "get_audit_logger",
]
