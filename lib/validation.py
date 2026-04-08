#!/usr/bin/env python3
"""
JSON Schema Validation for Ed4All Decision Capture

Validates decision records against defined schemas before writing.
Ensures all training data meets quality standards.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .constants import SCHEMAS_DIR, VALID_DECISION_TYPES

try:
    import jsonschema  # noqa: F401
    from jsonschema import Draft7Validator, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

# Schema paths
DECISION_SCHEMA_PATH = SCHEMAS_DIR / "decision_event_schema.json"
TRAINFORGE_SCHEMA_PATH = SCHEMAS_DIR / "trainforge_decision_schema.json"
SESSION_SCHEMA_PATH = SCHEMAS_DIR / "session_annotation_schema.json"

# Cache loaded schemas
_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}


def load_schema(schema_name: str) -> Dict[str, Any]:
    """
    Load and cache a JSON schema.

    Args:
        schema_name: "decision_event", "trainforge_decision", or "session_annotation"

    Returns:
        Parsed schema dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema is invalid JSON
    """
    if schema_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_name]

    schema_paths = {
        "decision_event": DECISION_SCHEMA_PATH,
        "trainforge_decision": TRAINFORGE_SCHEMA_PATH,
        "session_annotation": SESSION_SCHEMA_PATH
    }

    if schema_name not in schema_paths:
        raise ValueError(f"Unknown schema: {schema_name}. Valid options: {list(schema_paths.keys())}")

    schema_path = schema_paths[schema_name]
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_path) as f:
        schema = json.load(f)

    _SCHEMA_CACHE[schema_name] = schema
    return schema


def validate_decision(
    record: Dict[str, Any],
    tool: str = "courseforge",
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate a decision record against the appropriate schema.

    Args:
        record: The decision record to validate
        tool: "dart", "courseforge", "trainforge", or "orchestrator"
        strict: If True, raise exception on validation failure

    Returns:
        Tuple of (is_valid, list_of_issues)

    Raises:
        ValidationError: If strict=True and validation fails
    """
    if not JSONSCHEMA_AVAILABLE:
        # Fallback to basic validation if jsonschema not installed
        return _basic_validate(record)

    # Select schema based on tool
    schema_name = "trainforge_decision" if tool == "trainforge" else "decision_event"

    try:
        schema = load_schema(schema_name)
    except FileNotFoundError as e:
        return (False, [f"Schema not found: {e}"])

    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(record))

    if errors:
        issues = [_format_error(e) for e in errors]
        if strict:
            raise ValidationError(f"Decision validation failed: {issues}")
        return (False, issues)

    return (True, [])


def _format_error(error: Any) -> str:
    """Format a validation error into a readable string."""
    path = ".".join(str(p) for p in error.path) if error.path else "root"
    return f"{path}: {error.message}"


def _basic_validate(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Basic validation without jsonschema library.
    Checks required fields and minimum constraints.
    """
    issues = []

    # Required fields
    required = ["run_id", "timestamp", "operation", "decision_type", "decision", "rationale"]
    for field in required:
        if field not in record or not record[field]:
            issues.append(f"Missing required field: {field}")

    # Rationale minimum length
    rationale = record.get("rationale", "")
    if len(rationale) < 20:
        issues.append(f"Rationale too short: {len(rationale)} chars (minimum 20)")

    # Valid decision types (imported from constants)
    decision_type = record.get("decision_type", "")
    if decision_type and decision_type not in VALID_DECISION_TYPES:
        issues.append(f"Invalid decision_type: {decision_type}")

    # Confidence range
    confidence = record.get("confidence")
    if confidence is not None and (confidence < 0.0 or confidence > 1.0):
        issues.append(f"Confidence out of range: {confidence} (must be 0.0-1.0)")

    return (len(issues) == 0, issues)


def validate_phase_completeness(
    decisions: List[Dict[str, Any]],
    phase: str,
    tool: str = "courseforge"
) -> Dict[str, Any]:
    """
    Validate that a phase has sufficient decisions captured.

    Args:
        decisions: List of decision records for the phase
        phase: Phase name (e.g., "content-generator")
        tool: "dart", "courseforge", or "trainforge"

    Returns:
        Validation result dictionary
    """
    from .constants import MIN_DECISIONS_PER_PHASE

    min_required = MIN_DECISIONS_PER_PHASE.get(phase, 1)
    decision_count = len(decisions)

    result = {
        "valid": decision_count >= min_required,
        "phase": phase,
        "tool": tool,
        "decision_count": decision_count,
        "min_required": min_required,
        "issues": [],
        "quality_breakdown": {
            "exemplary": 0,
            "proficient": 0,
            "developing": 0,
            "inadequate": 0
        }
    }

    if decision_count < min_required:
        result["issues"].append(
            f"Insufficient decisions: {decision_count} < {min_required} required"
        )

    # Check individual decisions
    for i, decision in enumerate(decisions):
        is_valid, issues = validate_decision(decision, tool)
        if not is_valid:
            result["issues"].extend([f"Decision {i}: {issue}" for issue in issues])

        # Track quality breakdown
        quality = decision.get("metadata", {}).get("quality_level", "developing")
        if quality in result["quality_breakdown"]:
            result["quality_breakdown"][quality] += 1

    # Check for inadequate quality decisions
    inadequate = result["quality_breakdown"]["inadequate"]
    if inadequate > decision_count * 0.2:  # More than 20% inadequate
        result["issues"].append(
            f"Too many inadequate decisions: {inadequate}/{decision_count}"
        )
        result["valid"] = False

    return result


def validate_capture_file(
    filepath: Path,
    tool: str = "courseforge"
) -> Dict[str, Any]:
    """
    Validate a JSONL capture file.

    Args:
        filepath: Path to the .jsonl file
        tool: "dart", "courseforge", or "trainforge"

    Returns:
        Validation result dictionary
    """
    result = {
        "valid": True,
        "filepath": str(filepath),
        "decision_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "issues": []
    }

    if not filepath.exists():
        result["valid"] = False
        result["issues"].append(f"File not found: {filepath}")
        return result

    try:
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    result["decision_count"] += 1

                    is_valid, issues = validate_decision(record, tool)
                    if is_valid:
                        result["valid_count"] += 1
                    else:
                        result["invalid_count"] += 1
                        result["issues"].extend(
                            [f"Line {line_num}: {issue}" for issue in issues[:3]]  # Limit issues per line
                        )
                except json.JSONDecodeError as e:
                    result["issues"].append(f"Line {line_num}: Invalid JSON - {e}")
                    result["invalid_count"] += 1
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading file: {e}")
        return result

    # Overall validity
    if result["invalid_count"] > 0:
        result["valid"] = False

    return result


class CaptureValidator:
    """Validator class for batch validation of captures."""

    def __init__(self, tool: str = "courseforge"):
        self.tool = tool
        self.results: List[Dict[str, Any]] = []

    def validate_phase(
        self,
        course_code: str,
        phase: str
    ) -> Dict[str, Any]:
        """
        Validate all captures for a specific phase.

        Args:
            course_code: Course code (e.g., "MTH_101")
            phase: Phase name

        Returns:
            Validation result
        """
        from .constants import TRAINING_DIR

        phase_dir = TRAINING_DIR / self.tool / course_code / f"phase_{phase}"

        if not phase_dir.exists():
            return {
                "valid": False,
                "course_code": course_code,
                "phase": phase,
                "issues": [f"Phase directory not found: {phase_dir}"]
            }

        # Find all JSONL files
        jsonl_files = list(phase_dir.glob("decisions_*.jsonl"))
        if not jsonl_files:
            return {
                "valid": False,
                "course_code": course_code,
                "phase": phase,
                "issues": ["No capture files found"]
            }

        # Validate each file
        all_decisions = []
        file_results = []

        for jsonl_file in jsonl_files:
            file_result = validate_capture_file(jsonl_file, self.tool)
            file_results.append(file_result)

            # Load decisions for phase completeness check
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_decisions.append(json.loads(line))
            except Exception:
                pass

        # Check phase completeness
        phase_result = validate_phase_completeness(all_decisions, phase, self.tool)

        return {
            "valid": phase_result["valid"] and all(r["valid"] for r in file_results),
            "course_code": course_code,
            "phase": phase,
            "file_count": len(jsonl_files),
            "decision_count": len(all_decisions),
            "quality_breakdown": phase_result["quality_breakdown"],
            "file_results": file_results,
            "issues": phase_result["issues"]
        }

    def validate_course(self, course_code: str) -> Dict[str, Any]:
        """
        Validate all captures for a course.

        Args:
            course_code: Course code (e.g., "MTH_101")

        Returns:
            Full validation result for the course
        """
        from .constants import TRAINING_DIR

        course_dir = TRAINING_DIR / self.tool / course_code

        if not course_dir.exists():
            return {
                "valid": False,
                "course_code": course_code,
                "tool": self.tool,
                "issues": [f"Course directory not found: {course_dir}"]
            }

        # Find all phase directories
        phase_dirs = [d for d in course_dir.iterdir() if d.is_dir() and d.name.startswith("phase_")]

        result = {
            "valid": True,
            "course_code": course_code,
            "tool": self.tool,
            "phase_count": len(phase_dirs),
            "phase_results": {},
            "total_decisions": 0,
            "issues": []
        }

        for phase_dir in phase_dirs:
            phase_name = phase_dir.name.replace("phase_", "")
            phase_result = self.validate_phase(course_code, phase_name)
            result["phase_results"][phase_name] = phase_result
            result["total_decisions"] += phase_result.get("decision_count", 0)

            if not phase_result["valid"]:
                result["valid"] = False
                result["issues"].extend(
                    [f"{phase_name}: {issue}" for issue in phase_result.get("issues", [])]
                )

        return result


def check_jsonschema_available() -> bool:
    """Check if jsonschema library is available."""
    return JSONSCHEMA_AVAILABLE


def install_instructions() -> str:
    """Return installation instructions for jsonschema."""
    return "pip install jsonschema"
