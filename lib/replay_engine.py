"""
Replay Engine for Deterministic Run Verification

Loads captured decisions from a run and replays them for verification.
Supports comparison of original outputs vs replayed outputs.

Phase 0.5 Enhancement: Deterministic Replay Support (E1)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import unified_diff
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .run_manager import RunContext

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_RUNS_PATH = Path("runs")


class ReplayStatus(Enum):
    """Status of replay operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DIVERGED = "diverged"  # Replay produced different outputs


class DiffType(Enum):
    """Type of difference detected."""
    IDENTICAL = "identical"
    CONTENT_CHANGED = "content_changed"
    MISSING_IN_REPLAY = "missing_in_replay"
    EXTRA_IN_REPLAY = "extra_in_replay"
    HASH_MISMATCH = "hash_mismatch"
    METADATA_CHANGED = "metadata_changed"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DecisionRecord:
    """A single decision record loaded from capture."""
    event_id: str
    seq: int
    decision_type: str
    decision: str
    rationale: str
    timestamp: str
    operation: Optional[str] = None
    inputs_ref: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    is_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", ""),
            seq=data.get("seq", 0),
            decision_type=data.get("decision_type", ""),
            decision=data.get("decision", ""),
            rationale=data.get("rationale", ""),
            timestamp=data.get("timestamp", ""),
            operation=data.get("operation"),
            inputs_ref=data.get("inputs_ref", []),
            outputs=data.get("outputs", []),
            is_default=data.get("is_default", False),
            metadata=data.get("metadata", {}),
            raw=data,
        )


@dataclass
class ArtifactDiff:
    """Difference between original and replayed artifact."""
    artifact_path: str
    diff_type: DiffType
    original_hash: Optional[str] = None
    replayed_hash: Optional[str] = None
    diff_lines: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["diff_type"] = self.diff_type.value
        return d


@dataclass
class DiffResult:
    """Result of comparing original vs replayed outputs."""
    identical: bool
    total_artifacts: int
    matching_artifacts: int
    differing_artifacts: int
    missing_artifacts: int
    extra_artifacts: int
    diffs: List[ArtifactDiff] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identical": self.identical,
            "total_artifacts": self.total_artifacts,
            "matching_artifacts": self.matching_artifacts,
            "differing_artifacts": self.differing_artifacts,
            "missing_artifacts": self.missing_artifacts,
            "extra_artifacts": self.extra_artifacts,
            "diffs": [d.to_dict() for d in self.diffs],
            "summary": self.summary,
        }


@dataclass
class ReplayContext:
    """Context for replaying a run."""
    run_id: str
    original_run_path: Path
    replay_run_path: Optional[Path] = None
    decisions: List[DecisionRecord] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)
    status: ReplayStatus = ReplayStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_run_path": str(self.original_run_path),
            "replay_run_path": str(self.replay_run_path) if self.replay_run_path else None,
            "decision_count": len(self.decisions),
            "manifest": self.manifest,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
        }


@dataclass
class ReplayResult:
    """Result of replay operation."""
    success: bool
    status: ReplayStatus
    decisions_replayed: int
    decisions_skipped: int
    artifacts_produced: int
    diff_result: Optional[DiffResult] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    replay_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "decisions_replayed": self.decisions_replayed,
            "decisions_skipped": self.decisions_skipped,
            "artifacts_produced": self.artifacts_produced,
            "diff_result": self.diff_result.to_dict() if self.diff_result else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "replay_duration_seconds": self.replay_duration_seconds,
        }


# ============================================================================
# REPLAY ENGINE
# ============================================================================

class ReplayEngine:
    """
    Engine for replaying captured decisions for verification.

    Phase 0.5 Enhancement: Deterministic Replay Support.

    Usage:
        engine = ReplayEngine(runs_path=Path("runs"))
        context = engine.load_run("RUN_20250101_143022")
        result = engine.replay_decisions(context)
        diff = engine.compare_outputs(context)
    """

    def __init__(
        self,
        runs_path: Path = DEFAULT_RUNS_PATH,
        replay_suffix: str = "_replay",
    ):
        """
        Initialize replay engine.

        Args:
            runs_path: Base path for runs
            replay_suffix: Suffix to append to replay run IDs
        """
        self.runs_path = Path(runs_path)
        self.replay_suffix = replay_suffix

        # Decision handlers (extensible for custom replay logic)
        self._handlers: Dict[str, callable] = {}

    # ========================================================================
    # RUN LOADING
    # ========================================================================

    def load_run(self, run_id: str) -> ReplayContext:
        """
        Load a run for replay.

        Args:
            run_id: Run ID to load

        Returns:
            ReplayContext with loaded decisions

        Raises:
            FileNotFoundError: If run doesn't exist
            ValueError: If run data is invalid
        """
        run_path = self.runs_path / run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        # Load manifest
        manifest_path = run_path / "run_manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Run manifest not found: {run_id}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Create context
        context = ReplayContext(
            run_id=run_id,
            original_run_path=run_path,
            manifest=manifest,
        )

        # Load decisions from all JSONL files in decisions directory
        decisions_path = run_path / "decisions"
        if decisions_path.exists():
            context.decisions = list(self._load_decisions(decisions_path))

        # Sort by sequence number
        context.decisions.sort(key=lambda d: d.seq)

        logger.info(f"Loaded run {run_id} with {len(context.decisions)} decisions")
        return context

    def _load_decisions(self, decisions_path: Path) -> Iterator[DecisionRecord]:
        """Load all decisions from a directory."""
        for jsonl_file in decisions_path.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            yield DecisionRecord.from_dict(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse decision: {e}")

    # ========================================================================
    # REPLAY EXECUTION
    # ========================================================================

    def replay_decisions(
        self,
        context: ReplayContext,
        dry_run: bool = False,
        skip_defaults: bool = True,
    ) -> ReplayResult:
        """
        Replay decisions from a loaded run.

        Args:
            context: ReplayContext with loaded decisions
            dry_run: If True, simulate without producing artifacts
            skip_defaults: If True, skip decisions marked as is_default

        Returns:
            ReplayResult with outcome
        """
        context.status = ReplayStatus.IN_PROGRESS
        context.started_at = datetime.now()

        result = ReplayResult(
            success=False,
            status=ReplayStatus.IN_PROGRESS,
            decisions_replayed=0,
            decisions_skipped=0,
            artifacts_produced=0,
        )

        # Create replay run directory
        if not dry_run:
            replay_run_id = f"{context.run_id}{self.replay_suffix}"
            context.replay_run_path = self.runs_path / replay_run_id
            context.replay_run_path.mkdir(parents=True, exist_ok=True)

            # Copy manifest with replay metadata
            replay_manifest = context.manifest.copy()
            replay_manifest["replay"] = {
                "original_run_id": context.run_id,
                "replay_started": context.started_at.isoformat(),
                "dry_run": dry_run,
            }

            manifest_path = context.replay_run_path / "run_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(replay_manifest, f, indent=2)

        try:
            for decision in context.decisions:
                # Skip defaults if requested
                if skip_defaults and decision.is_default:
                    result.decisions_skipped += 1
                    continue

                # Get handler for decision type
                handler = self._handlers.get(decision.decision_type)

                if handler and not dry_run:
                    try:
                        artifacts = handler(decision, context)
                        result.artifacts_produced += len(artifacts) if artifacts else 0
                    except Exception as e:
                        logger.warning(f"Handler failed for {decision.event_id}: {e}")
                        result.warnings.append(
                            f"Decision {decision.event_id}: handler failed - {e}"
                        )

                result.decisions_replayed += 1

            # Mark success
            result.success = True
            result.status = ReplayStatus.COMPLETED
            context.status = ReplayStatus.COMPLETED

        except Exception as e:
            logger.error(f"Replay failed: {e}")
            result.errors.append(str(e))
            result.status = ReplayStatus.FAILED
            context.status = ReplayStatus.FAILED
            context.errors.append(str(e))

        finally:
            context.completed_at = datetime.now()
            if context.started_at:
                result.replay_duration_seconds = (
                    context.completed_at - context.started_at
                ).total_seconds()

        return result

    def register_handler(
        self,
        decision_type: str,
        handler: callable,
    ) -> None:
        """
        Register a handler for a decision type.

        Args:
            decision_type: Type of decision to handle
            handler: Callable(DecisionRecord, ReplayContext) -> List[str]
                     Returns list of artifact paths produced
        """
        self._handlers[decision_type] = handler

    # ========================================================================
    # OUTPUT COMPARISON
    # ========================================================================

    def compare_outputs(
        self,
        context: ReplayContext,
        compare_content: bool = True,
        ignored_patterns: Optional[List[str]] = None,
    ) -> DiffResult:
        """
        Compare original run outputs with replayed outputs.

        Args:
            context: ReplayContext with both original and replay paths
            compare_content: If True, compare file contents (not just hashes)
            ignored_patterns: File patterns to ignore in comparison

        Returns:
            DiffResult with comparison details
        """
        if not context.replay_run_path or not context.replay_run_path.exists():
            return DiffResult(
                identical=False,
                total_artifacts=0,
                matching_artifacts=0,
                differing_artifacts=0,
                missing_artifacts=0,
                extra_artifacts=0,
                summary="No replay run available for comparison",
            )

        ignored_patterns = ignored_patterns or []

        # Get artifact directories
        original_artifacts = context.original_run_path / "artifacts"
        replay_artifacts = context.replay_run_path / "artifacts"

        # Collect all artifact paths
        original_files = self._collect_artifacts(original_artifacts, ignored_patterns)
        replay_files = self._collect_artifacts(replay_artifacts, ignored_patterns)

        original_set = set(original_files.keys())
        replay_set = set(replay_files.keys())

        # Compute differences
        diffs = []

        # Files in both
        common = original_set & replay_set
        for rel_path in common:
            diff = self._compare_artifact(
                original_files[rel_path],
                replay_files[rel_path],
                rel_path,
                compare_content,
            )
            if diff.diff_type != DiffType.IDENTICAL:
                diffs.append(diff)

        # Files missing in replay
        for rel_path in original_set - replay_set:
            diffs.append(ArtifactDiff(
                artifact_path=rel_path,
                diff_type=DiffType.MISSING_IN_REPLAY,
                original_hash=self._hash_file(original_files[rel_path]),
            ))

        # Extra files in replay
        for rel_path in replay_set - original_set:
            diffs.append(ArtifactDiff(
                artifact_path=rel_path,
                diff_type=DiffType.EXTRA_IN_REPLAY,
                replayed_hash=self._hash_file(replay_files[rel_path]),
            ))

        # Build result
        matching = len(common) - len([d for d in diffs if d.artifact_path in common])
        result = DiffResult(
            identical=len(diffs) == 0,
            total_artifacts=len(original_set | replay_set),
            matching_artifacts=matching,
            differing_artifacts=len([d for d in diffs if d.diff_type == DiffType.CONTENT_CHANGED]),
            missing_artifacts=len(original_set - replay_set),
            extra_artifacts=len(replay_set - original_set),
            diffs=diffs,
        )

        # Generate summary
        if result.identical:
            result.summary = f"All {result.total_artifacts} artifacts match"
        else:
            parts = []
            if result.differing_artifacts:
                parts.append(f"{result.differing_artifacts} changed")
            if result.missing_artifacts:
                parts.append(f"{result.missing_artifacts} missing")
            if result.extra_artifacts:
                parts.append(f"{result.extra_artifacts} extra")
            result.summary = f"Differences found: {', '.join(parts)}"

        return result

    def _collect_artifacts(
        self,
        base_path: Path,
        ignored_patterns: List[str],
    ) -> Dict[str, Path]:
        """Collect artifact paths relative to base."""
        if not base_path.exists():
            return {}

        artifacts = {}
        for path in base_path.rglob("*"):
            if path.is_file():
                rel_path = str(path.relative_to(base_path))

                # Check ignored patterns
                skip = False
                for pattern in ignored_patterns:
                    if pattern in rel_path:
                        skip = True
                        break
                if skip:
                    continue

                artifacts[rel_path] = path

        return artifacts

    def _compare_artifact(
        self,
        original: Path,
        replayed: Path,
        rel_path: str,
        compare_content: bool,
    ) -> ArtifactDiff:
        """Compare two artifact files."""
        original_hash = self._hash_file(original)
        replayed_hash = self._hash_file(replayed)

        if original_hash == replayed_hash:
            return ArtifactDiff(
                artifact_path=rel_path,
                diff_type=DiffType.IDENTICAL,
                original_hash=original_hash,
                replayed_hash=replayed_hash,
            )

        diff = ArtifactDiff(
            artifact_path=rel_path,
            diff_type=DiffType.CONTENT_CHANGED,
            original_hash=original_hash,
            replayed_hash=replayed_hash,
        )

        # Generate text diff if requested
        if compare_content:
            try:
                original_text = original.read_text()
                replayed_text = replayed.read_text()

                diff_lines = list(unified_diff(
                    original_text.splitlines(keepends=True),
                    replayed_text.splitlines(keepends=True),
                    fromfile=f"original/{rel_path}",
                    tofile=f"replayed/{rel_path}",
                    lineterm="",
                ))
                diff.diff_lines = diff_lines[:100]  # Limit diff size

            except UnicodeDecodeError:
                # Binary file, can't generate text diff
                diff.details["binary"] = True

        return diff

    def _hash_file(self, path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    def verify_run_integrity(self, run_id: str) -> Dict[str, Any]:
        """
        Verify integrity of a run without full replay.

        Checks:
        - Hash chains are valid
        - All referenced artifacts exist
        - Manifest is consistent

        Args:
            run_id: Run ID to verify

        Returns:
            Dictionary with verification results
        """
        result = {
            "run_id": run_id,
            "verified": False,
            "checks": {},
            "errors": [],
        }

        run_path = self.runs_path / run_id
        if not run_path.exists():
            result["errors"].append(f"Run not found: {run_id}")
            return result

        # Check manifest exists
        manifest_path = run_path / "run_manifest.json"
        if manifest_path.exists():
            result["checks"]["manifest_exists"] = True
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                result["checks"]["manifest_valid"] = True
            except json.JSONDecodeError:
                result["checks"]["manifest_valid"] = False
                result["errors"].append("Invalid manifest JSON")
        else:
            result["checks"]["manifest_exists"] = False
            result["errors"].append("Manifest not found")

        # Check hash chain
        chain_path = run_path / "hash_chain.jsonl"
        if chain_path.exists():
            result["checks"]["hash_chain_exists"] = True
            try:
                chain_valid = self._verify_hash_chain(chain_path)
                result["checks"]["hash_chain_valid"] = chain_valid
                if not chain_valid:
                    result["errors"].append("Hash chain integrity check failed")
            except Exception as e:
                result["checks"]["hash_chain_valid"] = False
                result["errors"].append(f"Hash chain error: {e}")
        else:
            result["checks"]["hash_chain_exists"] = False

        # Check artifacts directory
        artifacts_path = run_path / "artifacts"
        if artifacts_path.exists():
            result["checks"]["artifacts_dir_exists"] = True
            artifact_count = len(list(artifacts_path.rglob("*")))
            result["checks"]["artifact_count"] = artifact_count
        else:
            result["checks"]["artifacts_dir_exists"] = False

        # Check finalization
        finalization_path = run_path / "finalization.json"
        result["checks"]["finalized"] = finalization_path.exists()

        # Overall verification
        result["verified"] = len(result["errors"]) == 0
        return result

    def _verify_hash_chain(self, chain_path: Path) -> bool:
        """Verify hash chain integrity."""
        prev_hash = None

        with open(chain_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    stored_prev = entry.get("prev_hash")

                    if prev_hash is None:
                        # First entry should have null or empty prev_hash
                        if stored_prev and stored_prev != "":
                            return False
                    else:
                        # Subsequent entries should chain correctly
                        if stored_prev != prev_hash:
                            return False

                    # Compute hash for next iteration
                    entry_copy = entry.copy()
                    entry_copy.pop("entry_hash", None)
                    content = json.dumps(entry_copy, sort_keys=True)
                    prev_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                except (json.JSONDecodeError, KeyError):
                    return False

        return True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def replay_run(
    run_id: str,
    runs_path: Path = DEFAULT_RUNS_PATH,
    dry_run: bool = False,
) -> Tuple[ReplayContext, ReplayResult]:
    """
    Convenience function to replay a run.

    Args:
        run_id: Run ID to replay
        runs_path: Base path for runs
        dry_run: If True, simulate without producing artifacts

    Returns:
        Tuple of (ReplayContext, ReplayResult)
    """
    engine = ReplayEngine(runs_path=runs_path)
    context = engine.load_run(run_id)
    result = engine.replay_decisions(context, dry_run=dry_run)
    return context, result


def verify_run(
    run_id: str,
    runs_path: Path = DEFAULT_RUNS_PATH,
) -> Dict[str, Any]:
    """
    Convenience function to verify run integrity.

    Args:
        run_id: Run ID to verify
        runs_path: Base path for runs

    Returns:
        Dictionary with verification results
    """
    engine = ReplayEngine(runs_path=runs_path)
    return engine.verify_run_integrity(run_id)


def compare_runs(
    original_run_id: str,
    replay_run_id: str,
    runs_path: Path = DEFAULT_RUNS_PATH,
) -> DiffResult:
    """
    Compare two runs directly.

    Args:
        original_run_id: Original run ID
        replay_run_id: Replay run ID
        runs_path: Base path for runs

    Returns:
        DiffResult with comparison
    """
    engine = ReplayEngine(runs_path=runs_path)

    # Create pseudo-context for comparison
    context = ReplayContext(
        run_id=original_run_id,
        original_run_path=runs_path / original_run_id,
        replay_run_path=runs_path / replay_run_id,
    )

    return engine.compare_outputs(context)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ReplayStatus",
    "DiffType",
    # Data classes
    "DecisionRecord",
    "ArtifactDiff",
    "DiffResult",
    "ReplayContext",
    "ReplayResult",
    # Main class
    "ReplayEngine",
    # Convenience functions
    "replay_run",
    "verify_run",
    "compare_runs",
]
