"""
Validation Gate Framework

Implements fail-closed validation gates for phase transitions.
Supports configurable severity, thresholds, and waiver capture.

Phase 0 Hardening - Requirement 3: Hard Validation Gates
"""

import importlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


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

    def __init__(self):
        """Initialize gate manager."""
        self._validators: Dict[str, Validator] = {}
        self._waivers: Dict[str, GateWaiver] = {}
        self._results_history: List[GateResult] = []

    # Allowlist of module prefixes permitted for validator imports.
    # Prevents arbitrary module loading (e.g., os, subprocess) via config.
    ALLOWED_VALIDATOR_PREFIXES = (
        "lib.validators.",
        "lib.leak_checker",
        "DART.pdf_converter.",
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
            result = validator.validate(inputs)
            result.gate_id = gate_config.gate_id

            # Apply threshold checks
            result = self._apply_thresholds(result, gate_config.threshold)

        except Exception as e:
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
