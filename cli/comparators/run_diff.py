"""
Run Diff Comparator

Compares two runs to identify differences in configuration, decisions, and outcomes.

Phase 0 Hardening - Requirement 9: CLI Integrity Checks
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

# Add project root to path
_COMPARATORS_DIR = Path(__file__).resolve().parent
_CLI_DIR = _COMPARATORS_DIR.parent
_PROJECT_ROOT = _CLI_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import STATE_PATH


@dataclass
class DiffItem:
    """Single difference item."""
    category: str
    key: str
    run_a_value: Any
    run_b_value: Any
    change_type: str  # "added", "removed", "modified", "unchanged"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "key": self.key,
            "run_a": self.run_a_value,
            "run_b": self.run_b_value,
            "change_type": self.change_type
        }


@dataclass
class DiffResult:
    """Result of comparing two runs."""
    run_a_id: str
    run_b_id: str
    config_diffs: List[DiffItem] = field(default_factory=list)
    decision_diffs: List[DiffItem] = field(default_factory=list)
    outcome_diffs: List[DiffItem] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_differences(self) -> int:
        return len(self.config_diffs) + len(self.decision_diffs) + len(self.outcome_diffs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_a": self.run_a_id,
            "run_b": self.run_b_id,
            "total_differences": self.total_differences,
            "config_diffs": [d.to_dict() for d in self.config_diffs],
            "decision_diffs": [d.to_dict() for d in self.decision_diffs],
            "outcome_diffs": [d.to_dict() for d in self.outcome_diffs],
            "summary": self.summary
        }

    def format(self, style: str = "text") -> str:
        """Format diff result for display."""
        if style == "json":
            return json.dumps(self.to_dict(), indent=2)
        elif style == "markdown":
            return self._format_markdown()
        else:
            return self._format_text()

    def _format_text(self) -> str:
        """Format as plain text."""
        lines = [
            f"Run Comparison: {self.run_a_id} vs {self.run_b_id}",
            "=" * 60,
            f"\nTotal Differences: {self.total_differences}",
        ]

        if self.config_diffs:
            lines.extend([
                "",
                "Configuration Differences",
                "-" * 40,
            ])
            for diff in self.config_diffs:
                symbol = self._change_symbol(diff.change_type)
                lines.append(f"  {symbol} {diff.key}")
                if diff.change_type == "modified":
                    lines.append(f"      A: {self._truncate(diff.run_a_value)}")
                    lines.append(f"      B: {self._truncate(diff.run_b_value)}")

        if self.decision_diffs:
            lines.extend([
                "",
                "Decision Pattern Differences",
                "-" * 40,
            ])
            for diff in self.decision_diffs:
                symbol = self._change_symbol(diff.change_type)
                lines.append(f"  {symbol} {diff.key}: {diff.run_a_value} -> {diff.run_b_value}")

        if self.outcome_diffs:
            lines.extend([
                "",
                "Outcome Differences",
                "-" * 40,
            ])
            for diff in self.outcome_diffs:
                symbol = self._change_symbol(diff.change_type)
                lines.append(f"  {symbol} {diff.key}")
                if diff.change_type == "modified":
                    lines.append(f"      A: {diff.run_a_value}")
                    lines.append(f"      B: {diff.run_b_value}")

        if self.summary:
            lines.extend([
                "",
                "Summary",
                "-" * 40,
            ])
            for key, value in self.summary.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def _format_markdown(self) -> str:
        """Format as Markdown."""
        lines = [
            f"# Run Comparison",
            "",
            f"**Run A:** `{self.run_a_id}`",
            f"**Run B:** `{self.run_b_id}`",
            "",
            f"**Total Differences:** {self.total_differences}",
        ]

        if self.config_diffs:
            lines.extend([
                "",
                "## Configuration Differences",
                "",
                "| Change | Key | Run A | Run B |",
                "|--------|-----|-------|-------|",
            ])
            for diff in self.config_diffs:
                symbol = self._change_symbol(diff.change_type)
                a_val = self._truncate(str(diff.run_a_value), 30) if diff.run_a_value else "-"
                b_val = self._truncate(str(diff.run_b_value), 30) if diff.run_b_value else "-"
                lines.append(f"| {symbol} | {diff.key} | {a_val} | {b_val} |")

        if self.decision_diffs:
            lines.extend([
                "",
                "## Decision Pattern Differences",
                "",
                "| Metric | Run A | Run B | Change |",
                "|--------|-------|-------|--------|",
            ])
            for diff in self.decision_diffs:
                change = ""
                if diff.run_a_value and diff.run_b_value:
                    try:
                        pct = ((diff.run_b_value - diff.run_a_value) / diff.run_a_value) * 100
                        change = f"{pct:+.1f}%"
                    except (TypeError, ZeroDivisionError):
                        change = "N/A"
                lines.append(f"| {diff.key} | {diff.run_a_value} | {diff.run_b_value} | {change} |")

        if self.outcome_diffs:
            lines.extend([
                "",
                "## Outcome Differences",
                "",
            ])
            for diff in self.outcome_diffs:
                lines.append(f"- **{diff.key}**: {diff.run_a_value} -> {diff.run_b_value}")

        if self.summary:
            lines.extend([
                "",
                "## Summary",
                "",
            ])
            for key, value in self.summary.items():
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def _change_symbol(self, change_type: str) -> str:
        """Get symbol for change type."""
        return {
            "added": "+",
            "removed": "-",
            "modified": "~",
            "unchanged": "=",
        }.get(change_type, "?")

    def _truncate(self, value: Any, max_len: int = 50) -> str:
        """Truncate value for display."""
        s = str(value)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s


class RunDiff:
    """Compares two runs."""

    def __init__(self, run_a_id: str, run_b_id: str, runs_root: Optional[Path] = None):
        """
        Initialize run diff comparator.

        Args:
            run_a_id: First run identifier
            run_b_id: Second run identifier
            runs_root: Root path for runs (defaults to state/runs)
        """
        self.run_a_id = run_a_id
        self.run_b_id = run_b_id
        self.runs_root = runs_root or STATE_PATH / "runs"
        self.run_a_path = self.runs_root / run_a_id
        self.run_b_path = self.runs_root / run_b_id

    def compare_all(self) -> DiffResult:
        """
        Compare all aspects of two runs.

        Returns:
            DiffResult with all differences
        """
        result = DiffResult(run_a_id=self.run_a_id, run_b_id=self.run_b_id)

        # Check both runs exist
        if not self.run_a_path.exists():
            result.summary["error"] = f"Run A not found: {self.run_a_id}"
            return result
        if not self.run_b_path.exists():
            result.summary["error"] = f"Run B not found: {self.run_b_id}"
            return result

        # Compare configs
        config_result = self.compare_configs()
        result.config_diffs = config_result.config_diffs

        # Compare decisions
        decision_result = self.compare_decisions()
        result.decision_diffs = decision_result.decision_diffs

        # Compare outcomes
        outcome_result = self.compare_outcomes()
        result.outcome_diffs = outcome_result.outcome_diffs

        # Generate summary
        result.summary = self._generate_summary(result)

        return result

    def compare_configs(self) -> DiffResult:
        """Compare configuration snapshots between runs."""
        result = DiffResult(run_a_id=self.run_a_id, run_b_id=self.run_b_id)

        # Load manifests
        manifest_a = self._load_json(self.run_a_path / "run_manifest.json")
        manifest_b = self._load_json(self.run_b_path / "run_manifest.json")

        if manifest_a is None or manifest_b is None:
            return result

        # Compare top-level manifest fields
        fields_to_compare = [
            "workflow_type", "operator", "git_commit", "git_dirty"
        ]
        for field_name in fields_to_compare:
            val_a = manifest_a.get(field_name)
            val_b = manifest_b.get(field_name)
            if val_a != val_b:
                result.config_diffs.append(DiffItem(
                    category="manifest",
                    key=field_name,
                    run_a_value=val_a,
                    run_b_value=val_b,
                    change_type="modified"
                ))

        # Compare config hashes
        hashes_a = manifest_a.get("config_hashes", {})
        hashes_b = manifest_b.get("config_hashes", {})

        all_files = set(hashes_a.keys()) | set(hashes_b.keys())
        for filename in sorted(all_files):
            hash_a = hashes_a.get(filename)
            hash_b = hashes_b.get(filename)

            if hash_a is None:
                result.config_diffs.append(DiffItem(
                    category="config_hash",
                    key=filename,
                    run_a_value=None,
                    run_b_value=hash_b,
                    change_type="added"
                ))
            elif hash_b is None:
                result.config_diffs.append(DiffItem(
                    category="config_hash",
                    key=filename,
                    run_a_value=hash_a,
                    run_b_value=None,
                    change_type="removed"
                ))
            elif hash_a != hash_b:
                result.config_diffs.append(DiffItem(
                    category="config_hash",
                    key=filename,
                    run_a_value=hash_a[:16] + "...",
                    run_b_value=hash_b[:16] + "...",
                    change_type="modified"
                ))

        # Compare workflow params
        params_a = manifest_a.get("workflow_params", {})
        params_b = manifest_b.get("workflow_params", {})

        self._compare_dicts(
            params_a, params_b, "workflow_param", result.config_diffs
        )

        return result

    def compare_decisions(self) -> DiffResult:
        """Compare decision patterns between runs."""
        result = DiffResult(run_a_id=self.run_a_id, run_b_id=self.run_b_id)

        # Load and count decisions
        decisions_a = self._load_decisions(self.run_a_path)
        decisions_b = self._load_decisions(self.run_b_path)

        # Compare total counts
        if decisions_a["total"] != decisions_b["total"]:
            result.decision_diffs.append(DiffItem(
                category="decision_count",
                key="total_decisions",
                run_a_value=decisions_a["total"],
                run_b_value=decisions_b["total"],
                change_type="modified"
            ))

        # Compare by decision type
        all_types = set(decisions_a["by_type"].keys()) | set(decisions_b["by_type"].keys())
        for dtype in sorted(all_types):
            count_a = decisions_a["by_type"].get(dtype, 0)
            count_b = decisions_b["by_type"].get(dtype, 0)

            if count_a != count_b:
                result.decision_diffs.append(DiffItem(
                    category="decision_type",
                    key=dtype,
                    run_a_value=count_a,
                    run_b_value=count_b,
                    change_type="added" if count_a == 0 else "removed" if count_b == 0 else "modified"
                ))

        # Compare rationale length statistics
        if decisions_a["avg_rationale_len"] != decisions_b["avg_rationale_len"]:
            result.decision_diffs.append(DiffItem(
                category="decision_quality",
                key="avg_rationale_length",
                run_a_value=round(decisions_a["avg_rationale_len"], 1),
                run_b_value=round(decisions_b["avg_rationale_len"], 1),
                change_type="modified"
            ))

        return result

    def compare_outcomes(self) -> DiffResult:
        """Compare run outcomes."""
        result = DiffResult(run_a_id=self.run_a_id, run_b_id=self.run_b_id)

        # Load manifests for status
        manifest_a = self._load_json(self.run_a_path / "run_manifest.json")
        manifest_b = self._load_json(self.run_b_path / "run_manifest.json")

        if manifest_a and manifest_b:
            status_a = manifest_a.get("status")
            status_b = manifest_b.get("status")
            if status_a != status_b:
                result.outcome_diffs.append(DiffItem(
                    category="outcome",
                    key="status",
                    run_a_value=status_a,
                    run_b_value=status_b,
                    change_type="modified"
                ))

        # Compare checkpoints
        checkpoints_a = self._load_checkpoints(self.run_a_path)
        checkpoints_b = self._load_checkpoints(self.run_b_path)

        phases_a = {cp["phase_name"]: cp["status"] for cp in checkpoints_a}
        phases_b = {cp["phase_name"]: cp["status"] for cp in checkpoints_b}

        all_phases = set(phases_a.keys()) | set(phases_b.keys())
        for phase in sorted(all_phases):
            status_a = phases_a.get(phase, "missing")
            status_b = phases_b.get(phase, "missing")

            if status_a != status_b:
                result.outcome_diffs.append(DiffItem(
                    category="phase_status",
                    key=phase,
                    run_a_value=status_a,
                    run_b_value=status_b,
                    change_type="modified"
                ))

        # Compare artifact counts
        artifacts_a = self._count_artifacts(self.run_a_path)
        artifacts_b = self._count_artifacts(self.run_b_path)

        if artifacts_a["count"] != artifacts_b["count"]:
            result.outcome_diffs.append(DiffItem(
                category="artifacts",
                key="artifact_count",
                run_a_value=artifacts_a["count"],
                run_b_value=artifacts_b["count"],
                change_type="modified"
            ))

        if abs(artifacts_a["size_mb"] - artifacts_b["size_mb"]) > 0.1:
            result.outcome_diffs.append(DiffItem(
                category="artifacts",
                key="artifact_size_mb",
                run_a_value=round(artifacts_a["size_mb"], 2),
                run_b_value=round(artifacts_b["size_mb"], 2),
                change_type="modified"
            ))

        # Compare validation results if present
        validation_a = self._load_json(self.run_a_path / "validation_results.json")
        validation_b = self._load_json(self.run_b_path / "validation_results.json")

        if validation_a and validation_b:
            score_a = validation_a.get("quality_score") or validation_a.get("score")
            score_b = validation_b.get("quality_score") or validation_b.get("score")

            if score_a != score_b:
                result.outcome_diffs.append(DiffItem(
                    category="quality",
                    key="quality_score",
                    run_a_value=score_a,
                    run_b_value=score_b,
                    change_type="modified"
                ))

        return result

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file safely."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _load_decisions(self, run_path: Path) -> Dict[str, Any]:
        """Load and aggregate decision statistics."""
        result = {
            "total": 0,
            "by_type": Counter(),
            "avg_rationale_len": 0,
        }

        decisions_dir = run_path / "decisions"
        if not decisions_dir.exists():
            return result

        rationale_lengths = []

        for decision_file in decisions_dir.glob("*.jsonl"):
            try:
                with open(decision_file) as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            result["total"] += 1
                            dtype = event.get("decision_type", "unknown")
                            result["by_type"][dtype] += 1

                            rationale = event.get("rationale", "")
                            if rationale:
                                rationale_lengths.append(len(rationale))
                        except json.JSONDecodeError:
                            pass
            except IOError:
                pass

        result["by_type"] = dict(result["by_type"])
        if rationale_lengths:
            result["avg_rationale_len"] = sum(rationale_lengths) / len(rationale_lengths)

        return result

    def _load_checkpoints(self, run_path: Path) -> List[Dict[str, Any]]:
        """Load checkpoint data."""
        checkpoints = []
        checkpoints_dir = run_path / "checkpoints"

        if not checkpoints_dir.exists():
            return checkpoints

        for cp_file in checkpoints_dir.glob("*_checkpoint.json"):
            data = self._load_json(cp_file)
            if data:
                checkpoints.append(data)

        return checkpoints

    def _count_artifacts(self, run_path: Path) -> Dict[str, Any]:
        """Count artifacts and total size."""
        result = {"count": 0, "size_mb": 0}
        artifacts_dir = run_path / "artifacts"

        if not artifacts_dir.exists():
            return result

        total_bytes = 0
        for artifact in artifacts_dir.rglob("*"):
            if artifact.is_file():
                result["count"] += 1
                try:
                    total_bytes += artifact.stat().st_size
                except OSError:
                    pass

        result["size_mb"] = total_bytes / (1024 * 1024)
        return result

    def _compare_dicts(
        self,
        dict_a: Dict,
        dict_b: Dict,
        category: str,
        diffs: List[DiffItem]
    ) -> None:
        """Compare two dictionaries and add differences to list."""
        all_keys = set(dict_a.keys()) | set(dict_b.keys())

        for key in sorted(all_keys):
            val_a = dict_a.get(key)
            val_b = dict_b.get(key)

            if val_a is None:
                diffs.append(DiffItem(
                    category=category,
                    key=key,
                    run_a_value=None,
                    run_b_value=val_b,
                    change_type="added"
                ))
            elif val_b is None:
                diffs.append(DiffItem(
                    category=category,
                    key=key,
                    run_a_value=val_a,
                    run_b_value=None,
                    change_type="removed"
                ))
            elif val_a != val_b:
                diffs.append(DiffItem(
                    category=category,
                    key=key,
                    run_a_value=val_a,
                    run_b_value=val_b,
                    change_type="modified"
                ))

    def _generate_summary(self, result: DiffResult) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "config_changes": len(result.config_diffs),
            "decision_changes": len(result.decision_diffs),
            "outcome_changes": len(result.outcome_diffs),
            "total_changes": result.total_differences,
            "runs_identical": result.total_differences == 0,
        }
