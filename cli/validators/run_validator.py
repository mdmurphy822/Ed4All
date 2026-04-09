"""
Run Validator

Validates run integrity including manifest, lockfile, hash chains, and artifacts.

Phase 0 Hardening - Requirement 9: CLI Integrity Checks
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
_VALIDATORS_DIR = Path(__file__).resolve().parent
_CLI_DIR = _VALIDATORS_DIR.parent
_PROJECT_ROOT = _CLI_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import STATE_PATH


@dataclass
class ValidationIssue:
    """Single validation issue."""
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    path: Optional[str] = None
    fixable: bool = False
    fix_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "path": self.path,
            "fixable": self.fixable,
            "fix_action": self.fix_action
        }


@dataclass
class ValidationResult:
    """Result of validation operation."""
    passed: bool
    run_id: str
    checked_files: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    fixed_count: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "run_id": self.run_id,
            "checked_files": self.checked_files,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "fixed_count": self.fixed_count,
            "issues": [i.to_dict() for i in self.issues]
        }


class RunValidator:
    """Validates run integrity."""

    def __init__(self, run_id: str, runs_root: Optional[Path] = None):
        """
        Initialize run validator.

        Args:
            run_id: Run identifier
            runs_root: Root path for runs (defaults to state/runs)
        """
        self.run_id = run_id
        self.runs_root = runs_root or STATE_PATH / "runs"
        self.run_path = self.runs_root / run_id

    def validate(self, fix: bool = False) -> ValidationResult:
        """
        Run all validation checks.

        Args:
            fix: Whether to attempt to fix fixable issues

        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(passed=True, run_id=self.run_id)

        # Check run directory exists
        if not self.run_path.exists():
            result.passed = False
            result.issues.append(ValidationIssue(
                severity="error",
                category="missing_run",
                message=f"Run directory not found: {self.run_path}"
            ))
            return result

        # Run all checks
        self._check_manifest(result)
        self._check_lockfile(result, fix)
        self._check_hash_chains(result)
        self._check_checkpoints(result)
        self._check_artifacts(result)
        self._check_decisions(result)

        result.passed = result.error_count == 0
        return result

    def _check_manifest(self, result: ValidationResult) -> None:
        """Verify run manifest exists and is valid."""
        manifest_path = self.run_path / "run_manifest.json"
        result.checked_files += 1

        if not manifest_path.exists():
            result.issues.append(ValidationIssue(
                severity="error",
                category="missing_manifest",
                message="Run manifest not found"
            ))
            return

        try:
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)

            # Validate required fields
            required_fields = ['run_id', 'created_at', 'workflow_type']
            for field in required_fields:
                if field not in manifest:
                    result.issues.append(ValidationIssue(
                        severity="error",
                        category="invalid_manifest",
                        message=f"Missing required field: {field}",
                        path=str(manifest_path)
                    ))

            # Verify run_id matches
            if manifest.get('run_id') != self.run_id:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    category="manifest_mismatch",
                    message=f"Manifest run_id doesn't match directory: {manifest.get('run_id')} vs {self.run_id}",
                    path=str(manifest_path)
                ))

            # Validate immutable flag
            if not manifest.get('immutable', True):
                result.issues.append(ValidationIssue(
                    severity="warning",
                    category="manifest_not_immutable",
                    message="Manifest should be marked immutable",
                    path=str(manifest_path)
                ))

        except json.JSONDecodeError as e:
            result.issues.append(ValidationIssue(
                severity="error",
                category="corrupt_manifest",
                message=f"Invalid JSON: {e}",
                path=str(manifest_path)
            ))

    def _check_lockfile(self, result: ValidationResult, fix: bool) -> None:
        """Verify config lockfile integrity."""
        lockfile_path = self.run_path / "config_lockfile.json"

        if not lockfile_path.exists():
            result.issues.append(ValidationIssue(
                severity="warning",
                category="missing_lockfile",
                message="Config lockfile not found (run may not have lockfile support)"
            ))
            return

        result.checked_files += 1

        try:
            with open(lockfile_path, encoding='utf-8') as f:
                lockfile = json.load(f)

            snapshot_dir = self.run_path / "config_snapshot"

            # Verify each config hash
            for filename, expected_hash in lockfile.get('config_hashes', {}).items():
                snapshot_path = snapshot_dir / filename
                result.checked_files += 1

                if not snapshot_path.exists():
                    result.issues.append(ValidationIssue(
                        severity="warning",
                        category="missing_snapshot",
                        message=f"Config snapshot missing: {filename}",
                        path=str(snapshot_path)
                    ))
                    continue

                # Verify hash
                try:
                    from lib.provenance import hash_file
                    actual_hash = hash_file(snapshot_path)
                    if actual_hash != expected_hash:
                        result.issues.append(ValidationIssue(
                            severity="error",
                            category="snapshot_modified",
                            message=f"Config snapshot hash mismatch for {filename}",
                            path=str(snapshot_path)
                        ))
                except ImportError:
                    pass  # Skip hash check if provenance module not available

        except json.JSONDecodeError as e:
            result.issues.append(ValidationIssue(
                severity="error",
                category="corrupt_lockfile",
                message=f"Invalid JSON: {e}",
                path=str(lockfile_path)
            ))

    def _check_hash_chains(self, result: ValidationResult) -> None:
        """Verify hash chain integrity for decision logs."""
        decisions_path = self.run_path / "decisions"

        if not decisions_path.exists():
            return

        try:
            from lib.hash_chain import HashChainedLog

            for chain_file in decisions_path.glob("*.jsonl"):
                result.checked_files += 1
                chain = HashChainedLog(chain_file)
                verification = chain.verify()

                if not verification.valid:
                    result.issues.append(ValidationIssue(
                        severity="error",
                        category="chain_broken",
                        message=f"Hash chain broken at seq {verification.break_at_seq}: {verification.error}",
                        path=str(chain_file)
                    ))
                elif verification.event_count > 0:
                    # Log successful verification as info (only in verbose)
                    pass

        except ImportError:
            result.issues.append(ValidationIssue(
                severity="info",
                category="skip_chain_check",
                message="Hash chain module not available, skipping verification"
            ))

    def _check_checkpoints(self, result: ValidationResult) -> None:
        """Verify checkpoint integrity."""
        checkpoints_path = self.run_path / "checkpoints"

        if not checkpoints_path.exists():
            return

        for checkpoint_file in checkpoints_path.glob("*_checkpoint.json"):
            result.checked_files += 1

            try:
                with open(checkpoint_file, encoding='utf-8') as f:
                    checkpoint = json.load(f)

                # Validate required fields
                required = ['run_id', 'phase_name', 'status']
                for field in required:
                    if field not in checkpoint:
                        result.issues.append(ValidationIssue(
                            severity="warning",
                            category="incomplete_checkpoint",
                            message=f"Checkpoint missing field: {field}",
                            path=str(checkpoint_file)
                        ))

                # Check for incomplete phases
                if checkpoint.get('status') == 'started':
                    result.issues.append(ValidationIssue(
                        severity="warning",
                        category="incomplete_phase",
                        message=f"Phase '{checkpoint.get('phase_name')}' never completed",
                        path=str(checkpoint_file)
                    ))

            except json.JSONDecodeError as e:
                result.issues.append(ValidationIssue(
                    severity="error",
                    category="corrupt_checkpoint",
                    message=f"Invalid JSON: {e}",
                    path=str(checkpoint_file)
                ))

    def _check_artifacts(self, result: ValidationResult) -> None:
        """Verify artifact integrity."""
        artifacts_path = self.run_path / "artifacts"

        if not artifacts_path.exists():
            return

        # Check for broken symlinks
        for artifact in artifacts_path.rglob("*"):
            if artifact.is_symlink():
                result.checked_files += 1
                target = artifact.resolve()

                if not target.exists():
                    result.issues.append(ValidationIssue(
                        severity="warning",
                        category="broken_symlink",
                        message="Artifact symlink target missing",
                        path=str(artifact),
                        fixable=True,
                        fix_action="remove_symlink"
                    ))

    def _check_decisions(self, result: ValidationResult) -> None:
        """Verify decision event schema compliance."""
        decisions_path = self.run_path / "decisions"

        if not decisions_path.exists():
            return

        # Check each JSONL file
        for decision_file in decisions_path.glob("*.jsonl"):
            if decision_file.name.startswith("chain_"):
                continue  # Skip chain files, handled separately

            result.checked_files += 1
            line_count = 0
            error_count = 0

            try:
                with open(decision_file, encoding='utf-8') as f:
                    for _line_num, line in enumerate(f, 1):
                        line_count += 1
                        try:
                            event = json.loads(line)

                            # Check required fields
                            if 'decision_type' not in event:
                                error_count += 1
                            if 'timestamp' not in event:
                                error_count += 1

                        except json.JSONDecodeError:
                            error_count += 1

                if error_count > 0:
                    result.issues.append(ValidationIssue(
                        severity="warning",
                        category="decision_errors",
                        message=f"{error_count}/{line_count} events have schema issues",
                        path=str(decision_file)
                    ))

            except Exception as e:
                result.issues.append(ValidationIssue(
                    severity="error",
                    category="decision_read_error",
                    message=str(e),
                    path=str(decision_file)
                ))

    def quick_check(self) -> bool:
        """
        Quick integrity check (spot check, not comprehensive).

        Returns:
            True if basic checks pass
        """
        # Check run directory exists
        if not self.run_path.exists():
            return False

        # Check manifest exists
        if not (self.run_path / "run_manifest.json").exists():
            return False

        # Check manifest is valid JSON
        try:
            with open(self.run_path / "run_manifest.json", encoding='utf-8') as f:
                json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        return True
