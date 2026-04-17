#!/usr/bin/env python3
"""
Claude Decision Capture Utility for Ed4All

Captures all Claude decision points during:
- DART PDF conversion
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

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import (
    MIN_DECISIONS_PER_PHASE,
    OPERATION_MAP,
    VALIDATE_DECISIONS,
)
from .libv2_storage import LibV2Storage
from .paths import TRAINING_DIR as LEGACY_TRAINING_DIR
from .quality import assess_decision_quality

# Phase 0 Hardening imports (graceful fallback if not available)
try:
    from .provenance import InputRef as ProvenanceInputRef  # noqa: F401
    from .provenance import OutputRef, create_input_ref, create_output_ref  # noqa: F401
    from .run_manager import HARDENED_MODE, get_current_run
    from .sequence_manager import generate_event_id, get_sequence_for_context
    HARDENING_AVAILABLE = True
except ImportError:
    HARDENING_AVAILABLE = False
    HARDENED_MODE = False

    def get_current_run():
        return None

    def get_sequence_for_context(run_id=None):
        return 0, f"EVT_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]}"

    def generate_event_id():
        return f"EVT_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]}"

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


# ADR-001 Contract 3: decision-type registry.
#
# Prior to Worker C this was a free-string field; the enum is established here
# (Worker C's first landing decision type is ``instruction_pair_synthesis``).
#
# Convention: add a new type in the same PR that first uses it in production.
# New types are ``snake_case`` and tool-prefixed when ambiguous. The registry
# is advisory rather than enforced — :func:`DecisionCapture.log_decision` does
# NOT raise on unknown types, because legacy callers across the tree emit many
# free-string types that have not yet been catalogued here. The list is the
# durable coordination artefact; enforcement can land in a follow-up PR once
# the legacy types are catalogued (tracked as FOLLOWUP-ADR001-5).
ALLOWED_DECISION_TYPES: tuple = (
    # Worker C (training-pair synthesis, landed in worker-c/training-pairs):
    "instruction_pair_synthesis",
    "preference_pair_generation",
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
            tool: "dart", "courseforge", or "trainforge"
            streaming: If True, write decisions immediately to disk (crash-safe)
            task_id: Phase 0 - Orchestrator task ID for cross-linking
        """
        self.course_code = course_code
        self.phase = phase
        self.tool = tool
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.streaming_mode = streaming
        self.task_id = task_id  # Phase 0: Cross-link to orchestrator

        # Phase 0 Hardening: Check for active run context
        self._run_context = get_current_run() if HARDENING_AVAILABLE else None

        # Stable IDs for ML training
        if self._run_context:
            # Use run_id from active run context
            self.run_id = self._run_context.run_id
        else:
            self.run_id = os.environ.get('RUN_ID', f"{tool}_{course_code}_{self.session_id}")

        self.course_id = course_code.replace(' ', '_').upper()  # Normalized: "MTH_101"
        self.module_id: Optional[str] = None  # Set per-module via set_module_context()
        self.artifact_id: Optional[str] = None  # Hash of produced content

        # Use LibV2 storage for training captures (primary location)
        self._storage = LibV2Storage(course_code, auto_create=True)
        self.output_dir = self._storage.get_training_capture_path(tool, phase)

        # Legacy training-captures directory (secondary location per CLAUDE.md spec)
        normalized_phase = phase.replace("_", "-")
        self.legacy_output_dir = LEGACY_TRAINING_DIR / tool / course_code / f"phase_{normalized_phase}"
        self.legacy_output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 0 Hardening: Also write to run-specific decisions path
        self._run_decisions_path: Optional[Path] = None
        self._run_stream_file = None
        if self._run_context:
            self._run_decisions_path = self._run_context.decisions_path / f"decisions_{tool}_{self.session_id}.jsonl"

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

    def _infer_operation(self, decision_type: str) -> str:
        """Infer operation from decision type for ML labeling."""
        return OPERATION_MAP.get(decision_type, f"decide_{decision_type}")

    def close(self):
        """Explicitly close all stream file handles."""
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
        seq, event_id = get_sequence_for_context(self.run_id if self._run_context else None)

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
            "ml_features": asdict(ml_features) if ml_features else {},
            "inputs_ref": [asdict(ref) for ref in (inputs_ref or [])],
            "prompt_ref": prompt_ref,
            "outputs": [asdict(out) for out in (outputs or [])],
            "outcome": asdict(outcome) if outcome else None,
            "metadata": {
                "rationale_length": rationale_length,
                "quality_level": quality_level,
                "hardening_version": "1.0.0" if HARDENING_AVAILABLE else None,
                **kwargs
            }
        }

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
        """Validate a decision record, adding issues to metadata if found."""
        if not VALIDATE_DECISIONS:
            return
        try:
            from .validation import validate_decision
            is_valid, issues = validate_decision(record, self.tool)
            if not is_valid:
                logger.warning("Decision validation issues: %s", issues)
                record["metadata"]["validation_issues"] = issues
        except ImportError:
            pass  # Validation module not available
        except Exception as e:
            logger.warning("Decision validation error: %s", e)

    def _write_to_streams(self, record: Dict[str, Any]) -> None:
        """Write a decision record to all configured stream locations."""
        if not self.streaming_mode:
            return

        line = json.dumps(record) + '\n'

        # Phase 0.5: Try WriteFacade first for atomic writes
        if self._write_with_facade(line):
            return

        # Fall back to legacy triple-write
        for fh, label in [
            (self._stream_file, "decision stream"),
            (self._legacy_stream_file, "legacy stream"),
            (self._run_stream_file, "run stream"),
        ]:
            if fh:
                try:
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())
                except OSError as e:
                    logger.warning("%s write failed: %s", label, e)

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
                "total_files": len(self.files_created)
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


class DARTDecisionCapture(DecisionCapture):
    """Specialized decision capture for DART conversions."""

    def __init__(self, course_code: str, pdf_name: str):
        super().__init__(course_code, "dart-conversion", "dart")
        self.pdf_name = pdf_name
        self.conversion_details: Dict[str, Any] = {}

    def log_conversion_start(self, pdf_path: str, options: Dict[str, Any]):
        """Log the start of a DART conversion."""
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
        """Log completion of a DART conversion."""
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
        """Save with DART-specific details using atomic write."""
        self.close()

        if filename is None:
            filename = f"dart_conversion_{self.pdf_name}_{self.session_id}.json"

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
                logger.warning("fsync failed on DART primary save: %s", e)
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
            logger.warning("Failed to save DART capture to legacy location: %s", e)

        return output_path


def create_capture(course_code: str, phase: str, tool: str = "courseforge") -> DecisionCapture:
    """Factory function to create a decision capture instance."""
    return DecisionCapture(course_code, phase, tool)


def create_dart_capture(course_code: str, pdf_name: str) -> DARTDecisionCapture:
    """Factory function to create a DART decision capture instance."""
    return DARTDecisionCapture(course_code, pdf_name)
