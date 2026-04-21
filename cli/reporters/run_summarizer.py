"""
Run Summarizer

Generates summary reports for runs.

Phase 0 Hardening - Requirement 9: CLI Integrity Checks
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
_REPORTERS_DIR = Path(__file__).resolve().parent
_CLI_DIR = _REPORTERS_DIR.parent
_PROJECT_ROOT = _CLI_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import STATE_PATH


@dataclass
class RunSummary:
    """Summary of a run."""
    run_id: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    workflow_type: Optional[str] = None
    status: str = "unknown"

    # Timing
    duration_seconds: Optional[float] = None

    # Phases
    phases: List[Dict[str, Any]] = field(default_factory=list)
    phases_completed: int = 0
    phases_failed: int = 0

    # Decisions
    total_decisions: int = 0
    decision_types: Dict[str, int] = field(default_factory=dict)

    # Quality
    validation_results: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None

    # Artifacts
    artifacts_count: int = 0
    artifacts_size_bytes: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "workflow_type": self.workflow_type,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "phases": {
                "total": len(self.phases),
                "completed": self.phases_completed,
                "failed": self.phases_failed,
                "details": self.phases
            },
            "decisions": {
                "total": self.total_decisions,
                "by_type": self.decision_types
            },
            "validation_results": self.validation_results,
            "quality_score": self.quality_score,
            "artifacts": {
                "count": self.artifacts_count,
                "size_bytes": self.artifacts_size_bytes
            },
            "errors": self.errors
        }


class RunSummarizer:
    """Generates summary reports for runs."""

    def __init__(self, run_id: str, runs_root: Optional[Path] = None):
        """
        Initialize summarizer.

        Args:
            run_id: Run identifier
            runs_root: Root path for runs (defaults to state/runs)
        """
        self.run_id = run_id
        self.runs_root = runs_root or STATE_PATH / "runs"
        self.run_path = self.runs_root / run_id

    def generate(self, format: str = "text") -> str:
        """
        Generate summary report.

        Args:
            format: Output format (text, json, markdown)

        Returns:
            Formatted report string
        """
        summary = self._collect_summary()

        if format == "json":
            return json.dumps(summary.to_dict(), indent=2)
        elif format == "markdown":
            return self._format_markdown(summary)
        else:
            return self._format_text(summary)

    def _collect_summary(self) -> RunSummary:
        """Collect all summary information."""
        summary = RunSummary(run_id=self.run_id)

        if not self.run_path.exists():
            summary.status = "not_found"
            summary.errors.append(f"Run directory not found: {self.run_path}")
            return summary

        # Load manifest
        self._load_manifest(summary)

        # Load checkpoints
        self._load_checkpoints(summary)

        # Count decisions
        self._count_decisions(summary)

        # Load validation results
        self._load_validation_results(summary)

        # Count artifacts
        self._count_artifacts(summary)

        return summary

    def _load_manifest(self, summary: RunSummary) -> None:
        """Load manifest information."""
        manifest_path = self.run_path / "run_manifest.json"

        if not manifest_path.exists():
            summary.errors.append("Manifest not found")
            return

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            summary.created_at = manifest.get("created_at")
            summary.workflow_type = manifest.get("workflow_type")
            summary.status = manifest.get("status", "completed")

            # Calculate duration if both timestamps present
            if manifest.get("created_at") and manifest.get("completed_at"):
                try:
                    start = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(manifest["completed_at"].replace("Z", "+00:00"))
                    summary.duration_seconds = (end - start).total_seconds()
                    summary.completed_at = manifest["completed_at"]
                except Exception:
                    pass

        except Exception as e:
            summary.errors.append(f"Failed to load manifest: {e}")

    def _load_checkpoints(self, summary: RunSummary) -> None:
        """Load checkpoint information."""
        checkpoints_dir = self.run_path / "checkpoints"

        if not checkpoints_dir.exists():
            return

        for checkpoint_file in checkpoints_dir.glob("*_checkpoint.json"):
            try:
                with open(checkpoint_file) as f:
                    checkpoint = json.load(f)

                phase_info = {
                    "name": checkpoint.get("phase_name"),
                    "status": checkpoint.get("status"),
                    "tasks_completed": len(checkpoint.get("tasks_completed", [])),
                    "tasks_failed": len(checkpoint.get("tasks_failed", [])),
                }

                summary.phases.append(phase_info)

                if checkpoint.get("status") == "completed":
                    summary.phases_completed += 1
                elif checkpoint.get("status") == "failed":
                    summary.phases_failed += 1

                # Extract validation results
                if checkpoint.get("validation_results"):
                    summary.validation_results[checkpoint["phase_name"]] = checkpoint["validation_results"]

            except Exception as e:
                summary.errors.append(f"Failed to load checkpoint {checkpoint_file.name}: {e}")

    def _count_decisions(self, summary: RunSummary) -> None:
        """Count decision events."""
        decisions_dir = self.run_path / "decisions"

        if not decisions_dir.exists():
            return

        decision_types = Counter()

        for decision_file in decisions_dir.glob("*.jsonl"):
            try:
                with open(decision_file) as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            summary.total_decisions += 1
                            decision_type = event.get("decision_type", "unknown")
                            decision_types[decision_type] += 1
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                summary.errors.append(f"Failed to read decisions {decision_file.name}: {e}")

        summary.decision_types = dict(decision_types.most_common())

    def _load_validation_results(self, summary: RunSummary) -> None:
        """Load validation results and calculate quality score."""
        # Look for validation results in various locations
        validation_files = [
            self.run_path / "validation_results.json",
            self.run_path / "quality_report.json",
        ]

        for vf in validation_files:
            if vf.exists():
                try:
                    with open(vf) as f:
                        results = json.load(f)
                    summary.validation_results["final"] = results

                    # Extract quality score if present. Wave 28f
                    # (FOLLOWUP-ADR001-2): read `overall_quality_score`
                    # — the key the writer actually emits — and keep
                    # the legacy `quality_score` / `score` fallbacks
                    # so older artifacts still surface a score.
                    if "overall_quality_score" in results:
                        summary.quality_score = results["overall_quality_score"]
                    elif "quality_score" in results:
                        summary.quality_score = results["quality_score"]
                    elif "score" in results:
                        summary.quality_score = results["score"]

                except Exception:
                    pass

    def _count_artifacts(self, summary: RunSummary) -> None:
        """Count and measure artifacts."""
        artifacts_dir = self.run_path / "artifacts"

        if not artifacts_dir.exists():
            return

        inaccessible = 0
        for artifact in artifacts_dir.rglob("*"):
            if artifact.is_file():
                summary.artifacts_count += 1
                try:
                    summary.artifacts_size_bytes += artifact.stat().st_size
                except OSError:
                    inaccessible += 1
        if inaccessible:
            summary.errors.append(f"Could not stat {inaccessible} artifact file(s)")

    def _format_text(self, summary: RunSummary) -> str:
        """Format summary as plain text."""
        lines = [
            f"Run Summary: {summary.run_id}",
            "=" * 60,
            "",
            f"Status: {summary.status}",
            f"Workflow: {summary.workflow_type or 'N/A'}",
            f"Created: {summary.created_at or 'N/A'}",
        ]

        if summary.duration_seconds:
            duration_mins = summary.duration_seconds / 60
            lines.append(f"Duration: {duration_mins:.1f} minutes")

        lines.extend([
            "",
            "Phases",
            "-" * 40,
            f"  Completed: {summary.phases_completed}",
            f"  Failed: {summary.phases_failed}",
        ])

        for phase in summary.phases:
            status_icon = "✓" if phase["status"] == "completed" else "✗" if phase["status"] == "failed" else "○"
            lines.append(f"  {status_icon} {phase['name']}: {phase['tasks_completed']} tasks")

        lines.extend([
            "",
            "Decisions",
            "-" * 40,
            f"  Total: {summary.total_decisions}",
        ])

        for dtype, count in list(summary.decision_types.items())[:5]:
            lines.append(f"    {dtype}: {count}")

        if summary.quality_score is not None:
            lines.extend([
                "",
                "Quality",
                "-" * 40,
                f"  Score: {summary.quality_score:.2f}",
            ])

        lines.extend([
            "",
            "Artifacts",
            "-" * 40,
            f"  Count: {summary.artifacts_count}",
            f"  Size: {summary.artifacts_size_bytes / 1024 / 1024:.2f} MB",
        ])

        if summary.errors:
            lines.extend([
                "",
                "Errors",
                "-" * 40,
            ])
            for error in summary.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)

    def _format_markdown(self, summary: RunSummary) -> str:
        """Format summary as Markdown."""
        lines = [
            f"# Run Summary: {summary.run_id}",
            "",
            "## Overview",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Status | {summary.status} |",
            f"| Workflow | {summary.workflow_type or 'N/A'} |",
            f"| Created | {summary.created_at or 'N/A'} |",
        ]

        if summary.duration_seconds:
            duration_mins = summary.duration_seconds / 60
            lines.append(f"| Duration | {duration_mins:.1f} minutes |")

        lines.extend([
            "",
            "## Phases",
            "",
            f"- Completed: {summary.phases_completed}",
            f"- Failed: {summary.phases_failed}",
            "",
            "| Phase | Status | Tasks |",
            "|-------|--------|-------|",
        ])

        for phase in summary.phases:
            status_icon = "✅" if phase["status"] == "completed" else "❌" if phase["status"] == "failed" else "⏳"
            lines.append(f"| {phase['name']} | {status_icon} {phase['status']} | {phase['tasks_completed']} |")

        lines.extend([
            "",
            "## Decisions",
            "",
            f"**Total:** {summary.total_decisions}",
            "",
            "| Type | Count |",
            "|------|-------|",
        ])

        for dtype, count in list(summary.decision_types.items())[:10]:
            lines.append(f"| {dtype} | {count} |")

        if summary.quality_score is not None:
            lines.extend([
                "",
                "## Quality",
                "",
                f"**Score:** {summary.quality_score:.2f}",
            ])

        lines.extend([
            "",
            "## Artifacts",
            "",
            f"- Count: {summary.artifacts_count}",
            f"- Size: {summary.artifacts_size_bytes / 1024 / 1024:.2f} MB",
        ])

        if summary.errors:
            lines.extend([
                "",
                "## Errors",
                "",
            ])
            for error in summary.errors:
                lines.append(f"- {error}")

        return "\n".join(lines)
