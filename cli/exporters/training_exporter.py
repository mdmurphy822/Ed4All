"""
Training Data Exporter

Exports decision capture data in various ML training formats.

Phase 0 Hardening - Requirement 9: CLI Integrity Checks
"""

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# Add project root to path
_EXPORTERS_DIR = Path(__file__).resolve().parent
_CLI_DIR = _EXPORTERS_DIR.parent
_PROJECT_ROOT = _CLI_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import STATE_PATH


class ExportFormat(Enum):
    """Supported export formats."""
    JSONL = "jsonl"      # Raw decision events
    ALPACA = "alpaca"    # Instruction format for fine-tuning
    OPENAI = "openai"    # OpenAI-compatible chat format
    DPO = "dpo"          # Direct Preference Optimization pairs


class QualityLevel(Enum):
    """Minimum quality thresholds."""
    EXEMPLARY = "exemplary"     # Best decisions only
    PROFICIENT = "proficient"   # Good quality decisions
    DEVELOPING = "developing"   # All non-rejected decisions


@dataclass
class ExportStats:
    """Statistics from export operation."""
    total_events: int = 0
    exported_events: int = 0
    filtered_events: int = 0
    by_decision_type: Dict[str, int] = field(default_factory=dict)
    by_quality: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "exported_events": self.exported_events,
            "filtered_events": self.filtered_events,
            "by_decision_type": self.by_decision_type,
            "by_quality": self.by_quality,
            "warnings": self.warnings
        }


class TrainingExporter:
    """Exports training data from runs."""

    # Minimum rationale length for quality filtering
    MIN_RATIONALE_LENGTH = 20

    # Quality assessment thresholds
    QUALITY_THRESHOLDS = {
        "exemplary": {
            "min_rationale_length": 100,
            "requires_alternatives": True,
            "requires_ml_features": True,
        },
        "proficient": {
            "min_rationale_length": 50,
            "requires_alternatives": False,
            "requires_ml_features": False,
        },
        "developing": {
            "min_rationale_length": 20,
            "requires_alternatives": False,
            "requires_ml_features": False,
        },
    }

    def __init__(self, run_id: str, runs_root: Optional[Path] = None):
        """
        Initialize training exporter.

        Args:
            run_id: Run identifier
            runs_root: Root path for runs (defaults to state/runs)
        """
        self.run_id = run_id
        self.runs_root = runs_root or STATE_PATH / "runs"
        self.run_path = self.runs_root / run_id

    def export(
        self,
        output_path: Path,
        format: str = "jsonl",
        min_quality: str = "proficient",
        decision_types: Optional[List[str]] = None,
        include_rejected: bool = False,
    ) -> ExportStats:
        """
        Export training data to file.

        Args:
            output_path: Path for output file
            format: Export format (jsonl, alpaca, openai, dpo)
            min_quality: Minimum quality level (exemplary, proficient, developing)
            decision_types: Filter to specific decision types (optional)
            include_rejected: Whether to include rejected/negative examples

        Returns:
            ExportStats with export statistics
        """
        stats = ExportStats()

        # Validate run exists
        if not self.run_path.exists():
            stats.warnings.append(f"Run not found: {self.run_id}")
            return stats

        # Get format handler
        format_enum = ExportFormat(format.lower())
        formatter = self._get_formatter(format_enum)

        # Open output file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as out_file:
            # Process all decision files
            for event in self._iter_decisions():
                stats.total_events += 1

                # Apply filters
                if not self._passes_filters(
                    event, min_quality, decision_types, include_rejected
                ):
                    stats.filtered_events += 1
                    continue

                # Track statistics
                dtype = event.get("decision_type", "unknown")
                stats.by_decision_type[dtype] = stats.by_decision_type.get(dtype, 0) + 1

                quality = self._assess_quality(event)
                stats.by_quality[quality] = stats.by_quality.get(quality, 0) + 1

                # Format and write
                formatted = formatter(event)
                if formatted:
                    out_file.write(formatted + "\n")
                    stats.exported_events += 1

        return stats

    def export_dpo_pairs(
        self,
        output_path: Path,
        min_quality: str = "proficient",
    ) -> ExportStats:
        """
        Export DPO (Direct Preference Optimization) training pairs.

        Creates pairs of (prompt, chosen, rejected) for preference learning.

        Args:
            output_path: Path for output file
            min_quality: Minimum quality for "chosen" examples

        Returns:
            ExportStats with export statistics
        """
        stats = ExportStats()

        if not self.run_path.exists():
            stats.warnings.append(f"Run not found: {self.run_id}")
            return stats

        # Collect events with alternatives
        events_with_alternatives = []

        for event in self._iter_decisions():
            stats.total_events += 1

            alternatives = event.get("alternatives_considered", [])
            if alternatives and self._assess_quality(event) != "developing":
                events_with_alternatives.append(event)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as out_file:
            for event in events_with_alternatives:
                pairs = self._create_dpo_pairs(event)
                for pair in pairs:
                    out_file.write(json.dumps(pair) + "\n")
                    stats.exported_events += 1

                dtype = event.get("decision_type", "unknown")
                stats.by_decision_type[dtype] = stats.by_decision_type.get(dtype, 0) + 1

        return stats

    def _iter_decisions(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all decision events in run."""
        decisions_dir = self.run_path / "decisions"

        if not decisions_dir.exists():
            return

        for decision_file in sorted(decisions_dir.glob("*.jsonl")):
            try:
                with open(decision_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            # Add source file for tracking
                            event["_source_file"] = decision_file.name
                            yield event
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

    def _passes_filters(
        self,
        event: Dict[str, Any],
        min_quality: str,
        decision_types: Optional[List[str]],
        include_rejected: bool,
    ) -> bool:
        """Check if event passes all filters."""
        # Decision type filter
        if decision_types:
            dtype = event.get("decision_type")
            if dtype not in decision_types:
                return False

        # Rejected filter
        is_rejected = event.get("is_rejected", False) or event.get("is_negative", False)
        if is_rejected and not include_rejected:
            return False

        # Quality filter
        quality = self._assess_quality(event)
        quality_order = ["exemplary", "proficient", "developing"]

        min_idx = quality_order.index(min_quality)
        actual_idx = quality_order.index(quality)

        if actual_idx > min_idx:
            return False

        return True

    def _assess_quality(self, event: Dict[str, Any]) -> str:
        """Assess quality level of a decision event."""
        rationale = event.get("rationale", "")
        rationale_len = len(rationale)

        has_alternatives = bool(event.get("alternatives_considered"))
        has_ml_features = bool(event.get("ml_features"))

        # Check exemplary
        threshold = self.QUALITY_THRESHOLDS["exemplary"]
        if (rationale_len >= threshold["min_rationale_length"] and
            (not threshold["requires_alternatives"] or has_alternatives) and
            (not threshold["requires_ml_features"] or has_ml_features)):
            return "exemplary"

        # Check proficient
        threshold = self.QUALITY_THRESHOLDS["proficient"]
        if rationale_len >= threshold["min_rationale_length"]:
            return "proficient"

        # Default to developing
        return "developing"

    def _get_formatter(self, format: ExportFormat) -> Callable[[Dict], Optional[str]]:
        """Get formatter function for export format."""
        formatters = {
            ExportFormat.JSONL: self._format_jsonl,
            ExportFormat.ALPACA: self._format_alpaca,
            ExportFormat.OPENAI: self._format_openai,
            ExportFormat.DPO: self._format_jsonl,  # DPO uses separate method
        }
        return formatters[format]

    def _format_jsonl(self, event: Dict[str, Any]) -> Optional[str]:
        """Format as raw JSONL."""
        # Remove internal fields
        export_event = {k: v for k, v in event.items() if not k.startswith("_")}
        return json.dumps(export_event)

    def _format_alpaca(self, event: Dict[str, Any]) -> Optional[str]:
        """Format as Alpaca instruction format."""
        decision_type = event.get("decision_type", "decision")
        decision = event.get("decision", "")
        rationale = event.get("rationale", "")
        context = event.get("context", {})

        # Build instruction
        instruction = f"Make a {decision_type} for educational content creation."

        # Build input from context
        input_parts = []
        if context.get("course_id"):
            input_parts.append(f"Course: {context['course_id']}")
        if context.get("week"):
            input_parts.append(f"Week: {context['week']}")
        if context.get("objective"):
            input_parts.append(f"Objective: {context['objective']}")
        if event.get("inputs_ref"):
            input_parts.append(f"Inputs: {len(event['inputs_ref'])} reference(s)")

        input_text = "\n".join(input_parts) if input_parts else ""

        # Build output
        output = f"Decision: {decision}\n\nRationale: {rationale}"

        alpaca_format = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }

        return json.dumps(alpaca_format)

    def _format_openai(self, event: Dict[str, Any]) -> Optional[str]:
        """Format as OpenAI chat completion format."""
        decision_type = event.get("decision_type", "decision")
        decision = event.get("decision", "")
        rationale = event.get("rationale", "")
        context = event.get("context", {})

        # System message
        system_msg = (
            "You are an expert educational content creator. "
            "You make well-reasoned decisions about course content, "
            "always providing clear rationale for your choices."
        )

        # Build user message from context
        user_parts = [f"Please make a {decision_type}."]

        if context.get("course_id"):
            user_parts.append(f"Course: {context['course_id']}")
        if context.get("objective"):
            user_parts.append(f"Learning objective: {context['objective']}")
        if event.get("alternatives_considered"):
            alts = [a.get("option", "") for a in event["alternatives_considered"]]
            user_parts.append(f"Consider these options: {', '.join(alts)}")

        user_msg = "\n".join(user_parts)

        # Assistant message
        assistant_msg = f"{decision}\n\nRationale: {rationale}"

        openai_format = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

        return json.dumps(openai_format)

    def _create_dpo_pairs(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create DPO pairs from event with alternatives."""
        pairs = []

        decision = event.get("decision", "")
        rationale = event.get("rationale", "")
        alternatives = event.get("alternatives_considered", [])

        if not alternatives:
            return pairs

        # Build prompt from context
        context = event.get("context", {})
        decision_type = event.get("decision_type", "decision")

        prompt_parts = [f"Make a {decision_type} for educational content."]
        if context.get("course_id"):
            prompt_parts.append(f"Course: {context['course_id']}")
        if context.get("objective"):
            prompt_parts.append(f"Objective: {context['objective']}")

        prompt = "\n".join(prompt_parts)

        # Chosen response is the actual decision
        chosen = f"{decision}\n\nRationale: {rationale}"

        # Create pair for each rejected alternative
        for alt in alternatives:
            alt_option = alt.get("option", "")
            rejection_reason = alt.get("rejected_because", "")

            # Build rejected response
            rejected = f"{alt_option}\n\n(This was rejected because: {rejection_reason})"

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "decision_type": decision_type,
                    "run_id": self.run_id,
                    "event_id": event.get("event_id"),
                }
            })

        return pairs


def list_exportable_runs(runs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List runs available for export.

    Args:
        runs_root: Root path for runs

    Returns:
        List of run info dictionaries
    """
    runs_root = runs_root or STATE_PATH / "runs"
    runs = []

    if not runs_root.exists():
        return runs

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        decisions_dir = run_dir / "decisions"
        if not decisions_dir.exists():
            continue

        # Count decisions
        decision_count = 0
        for df in decisions_dir.glob("*.jsonl"):
            try:
                with open(df) as f:
                    decision_count += sum(1 for _ in f)
            except OSError:
                pass

        if decision_count == 0:
            continue

        # Load manifest for metadata
        manifest_path = run_dir / "run_manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        runs.append({
            "run_id": run_dir.name,
            "created_at": manifest.get("created_at"),
            "workflow_type": manifest.get("workflow_type"),
            "decision_count": decision_count,
            "status": manifest.get("status", "unknown"),
        })

    return runs
