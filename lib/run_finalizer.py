"""
Run Finalizer - Run Finalization Enforcement

Provides finalization enforcement for workflow runs:
- Hash chain verification before close
- Artifact checksum generation
- Finalization report creation
- Immutability enforcement

Phase 0.5 Enhancement: Critical Enforcement (A1)
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hash_chain import HashChainedLog, VerificationResult


# ============================================================================
# CONSTANTS
# ============================================================================

FINALIZATION_SCHEMA_VERSION = "1.0.0"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChainVerificationSummary:
    """Summary of hash chain verification for a single chain."""
    chain_path: str
    valid: bool
    event_count: int
    head_hash: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactChecksum:
    """Checksum record for an artifact."""
    path: str
    hash_algorithm: str
    hash_value: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalizationReport:
    """Complete report of run finalization."""
    run_id: str
    finalized_at: str
    schema_version: str = FINALIZATION_SCHEMA_VERSION

    # Verification results
    all_chains_valid: bool = True
    chain_verifications: List[Dict[str, Any]] = field(default_factory=list)

    # Artifact checksums
    artifact_count: int = 0
    artifact_checksums: List[Dict[str, Any]] = field(default_factory=list)

    # Run statistics
    total_decisions: int = 0
    total_audit_events: int = 0

    # Finalization status
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinalizationReport":
        return cls(**data)


# ============================================================================
# RUN FINALIZER CLASS
# ============================================================================

class RunFinalizer:
    """
    Enforces run finalization with integrity checks.

    Performs:
    1. Hash chain verification for all chained logs
    2. Artifact checksum generation
    3. Finalization report creation
    4. Optional immutability enforcement
    """

    # Hash chains to verify (relative to run path)
    CHAIN_PATHS = [
        "audit/audit_chain.jsonl",
        "decisions/decisions_chain.jsonl",
    ]

    # Artifact directories to checksum
    ARTIFACT_DIRS = [
        "artifacts",
    ]

    def __init__(
        self,
        run_path: Path,
        run_id: str,
        verify_chains: bool = True,
        generate_checksums: bool = True,
        enforce_immutable: bool = False,
    ):
        """
        Initialize run finalizer.

        Args:
            run_path: Path to run directory
            run_id: Run ID
            verify_chains: Whether to verify hash chains
            generate_checksums: Whether to generate artifact checksums
            enforce_immutable: Whether to set immutable permissions
        """
        self.run_path = Path(run_path)
        self.run_id = run_id
        self.verify_chains = verify_chains
        self.generate_checksums = generate_checksums
        self.enforce_immutable = enforce_immutable

    def finalize(self) -> FinalizationReport:
        """
        Perform complete run finalization.

        Returns:
            FinalizationReport with verification results
        """
        report = FinalizationReport(
            run_id=self.run_id,
            finalized_at=datetime.now().isoformat(),
        )

        # Step 1: Verify all hash chains
        if self.verify_chains:
            self._verify_all_chains(report)

        # Step 2: Generate artifact checksums
        if self.generate_checksums:
            self._generate_all_checksums(report)

        # Step 3: Count decisions and audit events
        self._count_events(report)

        # Step 4: Write finalization report
        self._write_report(report)

        # Step 5: Enforce immutability if requested
        if self.enforce_immutable and report.success:
            self._enforce_immutability(report)

        return report

    def verify_only(self) -> FinalizationReport:
        """
        Verify run without finalizing (read-only check).

        Returns:
            FinalizationReport with verification results only
        """
        report = FinalizationReport(
            run_id=self.run_id,
            finalized_at=datetime.now().isoformat(),
        )

        self._verify_all_chains(report)
        self._count_events(report)

        return report

    def _verify_all_chains(self, report: FinalizationReport) -> None:
        """Verify all hash chains in the run."""
        all_valid = True

        for chain_rel_path in self.CHAIN_PATHS:
            chain_path = self.run_path / chain_rel_path

            if not chain_path.exists():
                # Missing chains are OK - they may not have been used
                continue

            try:
                chain = HashChainedLog(chain_path, auto_create=False)
                result = chain.verify()

                summary = ChainVerificationSummary(
                    chain_path=chain_rel_path,
                    valid=result.valid,
                    event_count=result.total_events,
                    head_hash=result.chain_head_hash,
                    error=result.error_message,
                )
                report.chain_verifications.append(summary.to_dict())

                if not result.valid:
                    all_valid = False
                    report.errors.append(
                        f"Hash chain invalid: {chain_rel_path} - {result.error_message}"
                    )

            except Exception as e:
                all_valid = False
                summary = ChainVerificationSummary(
                    chain_path=chain_rel_path,
                    valid=False,
                    event_count=0,
                    error=str(e),
                )
                report.chain_verifications.append(summary.to_dict())
                report.errors.append(f"Failed to verify chain {chain_rel_path}: {e}")

        report.all_chains_valid = all_valid
        if not all_valid:
            report.success = False

    def _generate_all_checksums(self, report: FinalizationReport) -> None:
        """Generate checksums for all artifacts."""
        for artifact_dir in self.ARTIFACT_DIRS:
            artifact_path = self.run_path / artifact_dir

            if not artifact_path.exists():
                continue

            for file_path in artifact_path.rglob("*"):
                if file_path.is_file():
                    try:
                        checksum = self._compute_file_checksum(file_path)
                        report.artifact_checksums.append(checksum.to_dict())
                        report.artifact_count += 1
                    except Exception as e:
                        report.warnings.append(
                            f"Failed to checksum {file_path.relative_to(self.run_path)}: {e}"
                        )

    def _compute_file_checksum(self, file_path: Path) -> ArtifactChecksum:
        """Compute checksum for a single file."""
        hasher = hashlib.sha256()
        size = 0

        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
                size += len(chunk)

        return ArtifactChecksum(
            path=str(file_path.relative_to(self.run_path)),
            hash_algorithm="sha256",
            hash_value=hasher.hexdigest(),
            size_bytes=size,
        )

    def _count_events(self, report: FinalizationReport) -> None:
        """Count decision and audit events."""
        # Count decisions
        decisions_dir = self.run_path / "decisions"
        if decisions_dir.exists():
            for jsonl_file in decisions_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, 'r') as f:
                        report.total_decisions += sum(1 for line in f if line.strip())
                except Exception:
                    pass

        # Count audit events
        audit_dir = self.run_path / "audit"
        if audit_dir.exists():
            for jsonl_file in audit_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, 'r') as f:
                        report.total_audit_events += sum(1 for line in f if line.strip())
                except Exception:
                    pass

    def _write_report(self, report: FinalizationReport) -> None:
        """Write finalization report to run directory."""
        report_path = self.run_path / "finalization_report.json"

        # Also write checksums to separate file if generated
        if report.artifact_checksums:
            checksums_path = self.run_path / "checksums.json"
            checksums_data = {
                "run_id": self.run_id,
                "generated_at": report.finalized_at,
                "hash_algorithm": "sha256",
                "checksums": report.artifact_checksums,
            }
            with open(checksums_path, 'w') as f:
                json.dump(checksums_data, f, indent=2)

        # Write full report
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

    def _enforce_immutability(self, report: FinalizationReport) -> None:
        """Set restrictive permissions on run directory."""
        try:
            # Make all files read-only
            for file_path in self.run_path.rglob("*"):
                if file_path.is_file():
                    os.chmod(file_path, 0o444)

            # Make directories read-execute only
            for dir_path in self.run_path.rglob("*"):
                if dir_path.is_dir():
                    os.chmod(dir_path, 0o555)

            # Make root directory read-execute only
            os.chmod(self.run_path, 0o555)

        except Exception as e:
            report.warnings.append(f"Failed to enforce immutability: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def finalize_run_with_verification(
    run_path: Path,
    run_id: str,
    verify_chains: bool = True,
    generate_checksums: bool = True,
) -> FinalizationReport:
    """
    Finalize a run with full verification.

    Args:
        run_path: Path to run directory
        run_id: Run ID
        verify_chains: Whether to verify hash chains
        generate_checksums: Whether to generate artifact checksums

    Returns:
        FinalizationReport
    """
    finalizer = RunFinalizer(
        run_path=run_path,
        run_id=run_id,
        verify_chains=verify_chains,
        generate_checksums=generate_checksums,
    )
    return finalizer.finalize()


def verify_run_integrity(run_path: Path, run_id: str) -> FinalizationReport:
    """
    Verify run integrity without modifying anything.

    Args:
        run_path: Path to run directory
        run_id: Run ID

    Returns:
        FinalizationReport with verification results
    """
    finalizer = RunFinalizer(
        run_path=run_path,
        run_id=run_id,
        verify_chains=True,
        generate_checksums=False,
    )
    return finalizer.verify_only()


def load_finalization_report(run_path: Path) -> Optional[FinalizationReport]:
    """
    Load existing finalization report.

    Args:
        run_path: Path to run directory

    Returns:
        FinalizationReport if exists, None otherwise
    """
    report_path = Path(run_path) / "finalization_report.json"

    if not report_path.exists():
        return None

    with open(report_path, 'r') as f:
        data = json.load(f)

    return FinalizationReport.from_dict(data)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "FINALIZATION_SCHEMA_VERSION",
    # Data classes
    "ChainVerificationSummary",
    "ArtifactChecksum",
    "FinalizationReport",
    # Main class
    "RunFinalizer",
    # Convenience functions
    "finalize_run_with_verification",
    "verify_run_integrity",
    "load_finalization_report",
]
