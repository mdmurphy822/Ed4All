#!/usr/bin/env python3
"""
Streaming Decision Capture for Ed4All Training Data Collection

Real-time decision streaming that:
- Writes decisions immediately to disk (no data loss on crash)
- Uses JSONL format (one JSON object per line)
- Supports validation of capture completeness
- Integrates with DecisionCapture class

Phase 0 Hardening additions:
- event_id: Unique identifier per event (EVT_{16-hex-chars})
- seq: Monotonic sequence number within run
- task_id: Cross-link to orchestrator tasks
- is_default: Flag for non-decisions (default value used)
- outputs: Artifact pointers (not blobs)
- Hash-chained event writing for tamper evidence
- Triple-write to run-specific location

Adapted from INTEGRATOR CURRICULUM streaming_capture.py
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

from .decision_capture import MLFeatures, InputRef, OutcomeSignals
from .constants import PROJECT_DIR, TRAINING_DIR, MIN_DECISIONS_PER_PHASE, OPERATION_MAP
from .libv2_storage import LibV2Storage, LIBV2_CATALOG
from .quality import assess_decision_quality
from .paths import TRAINING_DIR as LEGACY_TRAINING_DIR

# Phase 0 hardening imports (graceful fallback for backwards compatibility)
try:
    from .run_manager import get_current_run, RunContext
    from .sequence_manager import SequenceManager, generate_event_id
    from .hash_chain import HashChainedLog
    from .provenance import OutputRef
    HARDENING_AVAILABLE = True
except ImportError:
    HARDENING_AVAILABLE = False
    # Stub classes for type hints when hardening not available
    RunContext = None
    SequenceManager = None
    HashChainedLog = None
    OutputRef = None

    def get_current_run():
        return None

    def generate_event_id():
        import secrets
        return f"EVT_{secrets.token_hex(8)}"


@dataclass
class StreamingDecision:
    """A single decision record for streaming with ML-trainable fields."""
    # Phase 0 Hardening: Unique event identification
    event_id: Optional[str] = None  # EVT_{16-hex-chars}
    seq: Optional[int] = None  # Monotonic sequence within run
    task_id: Optional[str] = None  # Cross-link to orchestrator task (T-{8-hex})

    # Stable IDs
    run_id: str = ""
    course_id: str = ""
    module_id: Optional[str] = None
    artifact_id: Optional[str] = None

    # Tool and operation
    tool: str = ""
    operation: str = ""

    # Core fields
    timestamp: str = ""
    course_code: str = ""
    phase: str = ""
    decision_type: str = ""
    decision: str = ""
    rationale: str = ""
    alternatives_considered: List[str] = field(default_factory=list)
    context: Optional[str] = None
    confidence: Optional[float] = None

    # Phase 0 Hardening: Non-decision flag
    is_default: bool = False  # True if this captures a non-decision (default value used)

    # ML features
    ml_features: Dict[str, Any] = field(default_factory=dict)

    # Input references
    inputs_ref: List[Dict[str, Any]] = field(default_factory=list)
    prompt_ref: Optional[str] = None

    # Phase 0 Hardening: Output artifact pointers
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Outcome signals
    outcome: Optional[Dict[str, Any]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingDecisionCapture:
    """
    Real-time streaming decision capture.

    Writes each decision immediately to a JSONL file, flushing after each write.
    Crash-safe: all written decisions survive process termination.

    Phase 0 Hardening:
    - Supports event_id and seq for unique identification
    - Supports task_id for cross-linking to orchestrator tasks
    - Integrates with HashChainedLog for tamper-evident writing
    - Triple-write: primary (LibV2), legacy (training-captures), run-specific (state/runs/)

    Usage:
        with StreamingDecisionCapture("CIS_101", "content-generator", "courseforge") as capture:
            capture.log_decision(
                decision_type="content_structure",
                decision="Using 12-week module format",
                rationale="Matches semester timeline"
            )
            # Decision is immediately written to disk
    """

    def __init__(
        self,
        course_code: str,
        phase: str,
        tool: str = "courseforge",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,  # Phase 0: orchestrator task ID
        run_id: Optional[str] = None,  # Phase 0: explicit run ID
    ):
        """
        Initialize streaming capture.

        Args:
            course_code: Course code (e.g., "CIS_101")
            phase: Pipeline phase (e.g., "content-generator")
            tool: "dart", "courseforge", or "trainforge"
            session_id: Optional session identifier (auto-generated if not provided)
            task_id: Phase 0: Orchestrator task ID for cross-linking (T-{8-hex})
            run_id: Phase 0: Explicit run ID (overrides env var and auto-generation)
        """
        self.course_code = course_code
        self.phase = phase
        self.tool = tool
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Phase 0 Hardening: task_id for cross-linking
        self.task_id = task_id

        # Stable IDs for ML training
        # Priority: explicit run_id > env var > auto-generated
        self.run_id = run_id or os.environ.get('RUN_ID', f"{tool}_{course_code}_{self.session_id}")
        self.course_id = course_code.replace(' ', '_').upper()
        self.module_id: Optional[str] = None
        self.artifact_id: Optional[str] = None

        # Use LibV2 storage for training captures (primary location)
        self._storage = LibV2Storage(course_code, auto_create=True)
        self.output_dir = self._storage.get_training_capture_path(tool, phase)

        # Legacy training-captures directory (secondary location per CLAUDE.md spec)
        # Path: training-captures/{tool}/{course_code}/phase_{phase}/
        normalized_phase = phase.replace("_", "-")
        self.legacy_output_dir = LEGACY_TRAINING_DIR / tool / course_code / f"phase_{normalized_phase}"
        self.legacy_output_dir.mkdir(parents=True, exist_ok=True)

        # Streaming file path (JSONL format) - primary
        self.stream_path = self.output_dir / f"decisions_{self.session_id}.jsonl"
        # Legacy stream path - secondary
        self.legacy_stream_path = self.legacy_output_dir / f"decisions_{self.session_id}.jsonl"

        # Metadata file path
        self.meta_path = self.output_dir / f"meta_{self.session_id}.json"
        self.legacy_meta_path = self.legacy_output_dir / f"meta_{self.session_id}.json"

        # File handles (dual-write becomes triple-write with run-specific)
        self._file = None
        self._legacy_file = None
        self._decision_count = 0
        self._started_at = None

        # Phase 0 Hardening: Hash chain and run-specific output
        self._hash_chain: Optional[Any] = None  # HashChainedLog instance
        self._sequence_manager: Optional[Any] = None  # SequenceManager instance
        self._run_file = None  # Run-specific output file handle
        self._run_output_dir: Optional[Path] = None
        self._run_stream_path: Optional[Path] = None

        # Initialize hardening components if available
        if HARDENING_AVAILABLE:
            self._init_hardening()

    def _infer_operation(self, decision_type: str) -> str:
        """Infer operation from decision type for ML labeling."""
        return OPERATION_MAP.get(decision_type, f"decide_{decision_type}")

    def _init_hardening(self) -> None:
        """Initialize Phase 0 hardening components."""
        run_context = get_current_run()
        if run_context:
            # Use run-specific output directory
            self._run_output_dir = run_context.decisions_path / self.tool / self.phase
            self._run_output_dir.mkdir(parents=True, exist_ok=True)
            self._run_stream_path = self._run_output_dir / f"decisions_{self.session_id}.jsonl"

            # Initialize hash chain for tamper-evident logging
            chain_path = self._run_output_dir / f"chain_{self.session_id}.jsonl"
            self._hash_chain = HashChainedLog(chain_path)

            # Initialize sequence manager for monotonic event IDs
            self._sequence_manager = SequenceManager(run_context.run_path)

            # Update run_id from context if not explicitly set
            if self.run_id.startswith(f"{self.tool}_"):
                self.run_id = run_context.run_id

    def _get_next_sequence(self) -> Tuple[Optional[int], str]:
        """Get next sequence number and event ID.

        Returns:
            Tuple of (seq, event_id). seq may be None if hardening unavailable.
        """
        if self._sequence_manager:
            seq = self._sequence_manager.next_seq()
            event_id = generate_event_id()
            return seq, event_id
        else:
            # Fallback: generate event_id without sequence tracking
            event_id = generate_event_id()
            return None, event_id

    def set_module_context(
        self,
        week: int,
        module: int,
        artifact_hash: Optional[str] = None
    ):
        """Set the current module context for ID generation."""
        self.module_id = f"{self.course_id}_W{week:02d}_M{module:02d}"
        self.artifact_id = artifact_hash or hashlib.sha256(
            f"{self.module_id}_{self.session_id}".encode()
        ).hexdigest()[:12]

    def _ensure_open(self):
        """Ensure stream files are open (triple-write to all locations)."""
        if self._file is None:
            try:
                self._file = open(self.stream_path, 'a')
            except (IOError, OSError) as e:
                raise RuntimeError(f"Failed to open stream file {self.stream_path}: {e}")

            # Also open legacy file for dual-write
            try:
                self._legacy_file = open(self.legacy_stream_path, 'a')
            except (IOError, OSError) as e:
                import sys
                print(f"Warning: Failed to open legacy stream file {self.legacy_stream_path}: {e}", file=sys.stderr)
                self._legacy_file = None

            # Phase 0 Hardening: Open run-specific file for triple-write
            if self._run_stream_path:
                try:
                    self._run_file = open(self._run_stream_path, 'a')
                except (IOError, OSError) as e:
                    import sys
                    print(f"Warning: Failed to open run stream file {self._run_stream_path}: {e}", file=sys.stderr)
                    self._run_file = None

            self._started_at = datetime.now().isoformat()
            try:
                self._write_meta()
            except Exception as e:
                if self._file:
                    self._file.close()
                    self._file = None
                if self._legacy_file:
                    self._legacy_file.close()
                    self._legacy_file = None
                if self._run_file:
                    self._run_file.close()
                    self._run_file = None
                raise

    def _write_meta(self):
        """Write/update session metadata to all locations (triple-write)."""
        meta = {
            "course_code": self.course_code,
            "phase": self.phase,
            "tool": self.tool,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "task_id": self.task_id,  # Phase 0: orchestrator task cross-link
            "started_at": self._started_at,
            "last_updated": datetime.now().isoformat(),
            "decision_count": self._decision_count,
            "stream_file": str(self.stream_path),
            "legacy_stream_file": str(self.legacy_stream_path),
            "run_stream_file": str(self._run_stream_path) if self._run_stream_path else None,
            "hash_chain_enabled": self._hash_chain is not None,
            "status": "in_progress"
        }

        # Atomic write to primary location
        temp_path = self.meta_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(temp_path, self.meta_path)

        # Also write to legacy location
        try:
            legacy_temp_path = self.legacy_meta_path.with_suffix('.tmp')
            with open(legacy_temp_path, 'w') as f:
                json.dump(meta, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(legacy_temp_path, self.legacy_meta_path)
        except (IOError, OSError) as e:
            import sys
            print(f"Warning: Failed to write legacy meta: {e}", file=sys.stderr)

        # Phase 0 Hardening: Write to run-specific location
        if self._run_output_dir:
            try:
                run_meta_path = self._run_output_dir / f"meta_{self.session_id}.json"
                run_temp_path = run_meta_path.with_suffix('.tmp')
                with open(run_temp_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(run_temp_path, run_meta_path)
            except (IOError, OSError) as e:
                import sys
                print(f"Warning: Failed to write run meta: {e}", file=sys.stderr)

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
        outputs: Optional[List[Dict[str, Any]]] = None,  # Phase 0: output artifacts
        is_default: bool = False,  # Phase 0: non-decision flag
        **metadata
    ):
        """
        Log a decision immediately to disk with ML-trainable fields.

        Args:
            decision_type: Type of decision
            decision: The decision made
            rationale: Why this decision was made (minimum 20 chars recommended)
            operation: Explicit action label (inferred if not provided)
            alternatives_considered: Other options that were considered
            context: Additional context
            confidence: Confidence level (0.0 to 1.0)
            ml_features: Categorical ML fields
            inputs_ref: References to source materials
            prompt_ref: Prompt template version/hash
            outcome: Outcome signals for preference training
            outputs: Phase 0: List of output artifact references (path, hash, type)
            is_default: Phase 0: True if this captures a non-decision (default value used)
            **metadata: Additional key-value pairs
        """
        self._ensure_open()

        # Phase 0 Hardening: Get sequence number and event ID
        seq, event_id = self._get_next_sequence()

        # Rationale quality assessment
        rationale_length = len(rationale) if rationale else 0
        if rationale_length < 20 and decision_type not in ('prompt_response', 'file_creation', 'source_usage'):
            import sys
            print(f"Warning: Short rationale ({rationale_length} chars) for {decision_type}", file=sys.stderr)

        # Assess decision quality using centralized quality module
        quality_level = assess_decision_quality(rationale, inputs_ref, alternatives_considered, decision_type)

        metadata['rationale_length'] = rationale_length
        metadata['quality_level'] = quality_level

        record = StreamingDecision(
            # Phase 0 Hardening fields
            event_id=event_id,
            seq=seq,
            task_id=self.task_id,
            is_default=is_default,
            outputs=outputs or [],
            # Stable IDs
            run_id=self.run_id,
            course_id=self.course_id,
            module_id=self.module_id,
            artifact_id=self.artifact_id,
            # Tool and operation
            tool=self.tool,
            operation=operation or self._infer_operation(decision_type),
            # Core fields
            timestamp=datetime.now().isoformat(),
            course_code=self.course_code,
            phase=self.phase,
            decision_type=decision_type,
            decision=decision,
            rationale=rationale,
            alternatives_considered=alternatives_considered or [],
            context=context,
            confidence=confidence,
            # ML features and references
            ml_features=asdict(ml_features) if ml_features else {},
            inputs_ref=[asdict(ref) for ref in (inputs_ref or [])],
            prompt_ref=prompt_ref,
            outcome=asdict(outcome) if outcome else None,
            metadata=metadata
        )

        # Serialize record
        record_dict = asdict(record)
        line = json.dumps(record_dict) + '\n'

        # Primary location (LibV2)
        self._file.write(line)
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError as e:
            import sys
            print(f"Warning: fsync failed (primary): {e}", file=sys.stderr)

        # Legacy location (training-captures)
        if self._legacy_file:
            try:
                self._legacy_file.write(line)
                self._legacy_file.flush()
                os.fsync(self._legacy_file.fileno())
            except OSError as e:
                import sys
                print(f"Warning: legacy write failed: {e}", file=sys.stderr)

        # Phase 0 Hardening: Run-specific location (state/runs/{run_id}/decisions/)
        if self._run_file:
            try:
                self._run_file.write(line)
                self._run_file.flush()
                os.fsync(self._run_file.fileno())
            except OSError as e:
                import sys
                print(f"Warning: run-specific write failed: {e}", file=sys.stderr)

        # Phase 0 Hardening: Write to hash chain for tamper evidence
        if self._hash_chain:
            try:
                self._hash_chain.append(record_dict)
            except Exception as e:
                import sys
                print(f"Warning: hash chain append failed: {e}", file=sys.stderr)

        self._decision_count += 1

        # Update metadata periodically
        if self._decision_count % 5 == 0:
            self._write_meta()

        return event_id  # Phase 0: Return event_id for cross-referencing

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
        purpose: str = "",
        model: str = "claude",
        tokens: Optional[int] = None
    ):
        """Log a prompt/response pair for SFT training."""
        self.log_decision(
            decision_type="prompt_response",
            decision=response[:500] if len(response) > 500 else response,
            rationale=purpose,
            context=prompt[:500] if len(prompt) > 500 else prompt,
            full_prompt=prompt,
            full_response=response,
            prompt_length=len(prompt),
            response_length=len(response),
            model=model,
            tokens=tokens
        )

    def log_file_created(self, filepath: str, content_summary: str = ""):
        """Log a file creation as a decision."""
        self.log_decision(
            decision_type="file_creation",
            decision=f"Created file: {filepath}",
            rationale=content_summary,
            filepath=filepath
        )

    def log_source_used(self, source_type: str, source_path: str, usage: str = ""):
        """Log source material usage."""
        self.log_decision(
            decision_type="source_usage",
            decision=f"Used {source_type}: {source_path}",
            rationale=usage,
            source_type=source_type,
            source_path=source_path
        )

    def log_artifact_created(
        self,
        artifact_type: str,
        path: str,
        content_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        rationale: str = "Generated as part of workflow",
        **additional_metadata
    ) -> str:
        """
        Log creation of an output artifact with provenance tracking.

        Phase 0 Hardening: Records artifact with hash for provenance chain.

        Args:
            artifact_type: Type of artifact (html, imscc, assessment, etc.)
            path: File path relative to run artifacts
            content_hash: SHA-256 hash of content (computed if not provided)
            size_bytes: Size in bytes (computed if not provided)
            rationale: Why this artifact was created
            **additional_metadata: Additional metadata for the artifact

        Returns:
            event_id of the logged decision
        """
        # Compute hash and size if not provided and file exists
        artifact_path = Path(path)
        if artifact_path.exists():
            if content_hash is None:
                content_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if size_bytes is None:
                size_bytes = artifact_path.stat().st_size

        output_ref = {
            "artifact_type": artifact_type,
            "path": str(path),
            "content_hash": content_hash,
            "hash_algorithm": "sha256",
            "size_bytes": size_bytes,
            **additional_metadata
        }

        return self.log_decision(
            decision_type="file_creation",
            decision=f"Created {artifact_type}: {path}",
            rationale=rationale,
            outputs=[output_ref],
            filepath=str(path),
            artifact_type=artifact_type
        )

    def log_default_used(
        self,
        decision_type: str,
        default_value: str,
        rationale: str = "Used default value as no alternative was specified"
    ) -> str:
        """
        Log when a default value is used (non-decision capture).

        Phase 0 Hardening: Captures non-decisions for completeness.

        Args:
            decision_type: Category of decision
            default_value: The default value that was used
            rationale: Why the default was acceptable

        Returns:
            event_id of the logged decision
        """
        return self.log_decision(
            decision_type=decision_type,
            decision=f"Used default: {default_value}",
            rationale=rationale,
            is_default=True
        )

    def verify_hash_chain(self) -> Optional[Dict[str, Any]]:
        """
        Verify the integrity of the hash chain.

        Phase 0 Hardening: Returns verification result or None if no chain.

        Returns:
            Dict with 'valid', 'length', 'error' (if any) or None if no chain
        """
        if not self._hash_chain:
            return None

        try:
            verification = self._hash_chain.verify()
            return {
                "valid": verification.valid,
                "length": verification.length,
                "error": verification.error if not verification.valid else None
            }
        except Exception as e:
            return {
                "valid": False,
                "length": 0,
                "error": str(e)
            }

    def finalize(self, status: str = "complete"):
        """Finalize the capture session (close all file handles)."""
        # Close primary file
        if self._file:
            self._file.close()
            self._file = None

        # Close legacy file
        if self._legacy_file:
            self._legacy_file.close()
            self._legacy_file = None

        # Phase 0 Hardening: Close run-specific file
        if self._run_file:
            self._run_file.close()
            self._run_file = None

        # Phase 0 Hardening: Verify hash chain integrity
        hash_chain_valid = None
        hash_chain_length = None
        if self._hash_chain:
            try:
                verification = self._hash_chain.verify()
                hash_chain_valid = verification.valid
                hash_chain_length = verification.length
            except Exception:
                hash_chain_valid = False

        # Initialize meta with defaults in case file doesn't exist or is corrupt
        meta = {
            "course_code": self.course_code,
            "phase": self.phase,
            "tool": self.tool,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "decision_count": self._decision_count,
            "status": status,
            "stream_file": str(self.stream_path),
            "legacy_stream_file": str(self.legacy_stream_path),
            "run_stream_file": str(self._run_stream_path) if self._run_stream_path else None,
            # Phase 0 Hardening: Hash chain verification results
            "hash_chain_enabled": self._hash_chain is not None,
            "hash_chain_valid": hash_chain_valid,
            "hash_chain_length": hash_chain_length,
        }

        # Try to load existing meta to preserve started_at
        if self.meta_path.exists():
            try:
                with open(self.meta_path, 'r') as f:
                    existing_meta = json.load(f)
                    meta.update(existing_meta)
            except (json.JSONDecodeError, IOError):
                pass  # Use default meta if file is corrupt

        meta["completed_at"] = datetime.now().isoformat()
        meta["status"] = status
        meta["decision_count"] = self._decision_count
        meta["hash_chain_valid"] = hash_chain_valid
        meta["hash_chain_length"] = hash_chain_length

        # Write to primary location
        temp_path = self.meta_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(meta, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(temp_path, self.meta_path)
        except (IOError, OSError):
            pass  # Best effort - don't fail finalization on meta write error

        # Write to legacy location
        try:
            legacy_temp_path = self.legacy_meta_path.with_suffix('.tmp')
            with open(legacy_temp_path, 'w') as f:
                json.dump(meta, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(legacy_temp_path, self.legacy_meta_path)
        except (IOError, OSError):
            pass  # Best effort

        # Phase 0 Hardening: Write to run-specific location
        if self._run_output_dir:
            try:
                run_meta_path = self._run_output_dir / f"meta_{self.session_id}.json"
                run_temp_path = run_meta_path.with_suffix('.tmp')
                with open(run_temp_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(run_temp_path, run_meta_path)
            except (IOError, OSError):
                pass  # Best effort

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "complete"
        self.finalize(status)
        return False


class CaptureValidator:
    """Validates that decision capture is complete and sufficient."""

    @staticmethod
    def validate_phase(course_code: str, phase: str, tool: str = "courseforge") -> Dict[str, Any]:
        """Validate that a phase has sufficient decision capture."""
        storage = LibV2Storage(course_code)
        phase_dir = storage.get_training_capture_path(tool, phase)

        result = {
            "valid": False,
            "course_code": course_code,
            "phase": phase,
            "tool": tool,
            "decision_count": 0,
            "issues": []
        }

        if not phase_dir.exists():
            result["issues"].append(f"No capture directory: {phase_dir}")
            return result

        jsonl_files = list(phase_dir.glob("decisions_*.jsonl"))
        if not jsonl_files:
            result["issues"].append("No decision stream files found")
            return result

        total_decisions = 0
        decision_types = set()

        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            decision = json.loads(line)
                            total_decisions += 1
                            decision_types.add(decision.get("decision_type", "unknown"))
            except (json.JSONDecodeError, IOError) as e:
                result["issues"].append(f"Error reading {jsonl_file.name}: {e}")

        result["decision_count"] = total_decisions
        result["decision_types"] = list(decision_types)

        min_required = MIN_DECISIONS_PER_PHASE.get(phase, 1)
        if total_decisions < min_required:
            result["issues"].append(
                f"Insufficient decisions: {total_decisions} < {min_required} required"
            )

        if phase == "content-generator":
            if "file_creation" not in decision_types:
                result["issues"].append("No file creation decisions logged")
            if "prompt_response" not in decision_types:
                result["issues"].append("No prompt/response pairs logged")

        result["valid"] = len(result["issues"]) == 0
        return result

    @staticmethod
    def validate_course(course_code: str, tool: str = "courseforge") -> Dict[str, Any]:
        """Validate all phases for a course."""
        storage = LibV2Storage(course_code)
        course_dir = storage.training_path / tool

        result = {
            "valid": True,
            "course_code": course_code,
            "tool": tool,
            "phases": {},
            "total_decisions": 0
        }

        if not course_dir.exists():
            result["valid"] = False
            result["error"] = f"No training annotations for {course_code}"
            return result

        for phase_dir in course_dir.glob("phase_*"):
            phase = phase_dir.name.replace("phase_", "")
            phase_result = CaptureValidator.validate_phase(course_code, phase, tool)
            result["phases"][phase] = phase_result
            result["total_decisions"] += phase_result["decision_count"]

            if not phase_result["valid"]:
                result["valid"] = False

        return result


def create_streaming_capture(
    course_code: str,
    phase: str,
    tool: str = "courseforge",
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,  # Phase 0: orchestrator task ID
    run_id: Optional[str] = None,   # Phase 0: explicit run ID
) -> StreamingDecisionCapture:
    """
    Factory function to create a streaming capture instance.

    Args:
        course_code: Course code (e.g., "CIS_101")
        phase: Pipeline phase (e.g., "content-generator")
        tool: "dart", "courseforge", or "trainforge"
        session_id: Optional session identifier (auto-generated if not provided)
        task_id: Phase 0: Orchestrator task ID for cross-linking (T-{8-hex})
        run_id: Phase 0: Explicit run ID (overrides env var and auto-generation)

    Returns:
        StreamingDecisionCapture instance ready for use
    """
    return StreamingDecisionCapture(
        course_code=course_code,
        phase=phase,
        tool=tool,
        session_id=session_id,
        task_id=task_id,
        run_id=run_id
    )


def validate_phase_capture(course_code: str, phase: str, tool: str = "courseforge") -> bool:
    """Quick validation check for phase completion."""
    result = CaptureValidator.validate_phase(course_code, phase, tool)
    return result["valid"]


def get_capture_stats(course_code: str, tool: str = "courseforge") -> Dict[str, Any]:
    """Get capture statistics for a course."""
    return CaptureValidator.validate_course(course_code, tool)
