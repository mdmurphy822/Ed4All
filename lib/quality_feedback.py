"""
Quality Feedback System

Enables downstream pipeline stages (Trainforge, LibV2) to report quality
issues back upstream. When Trainforge generates low-quality questions from
a module's content, this feedback is logged so the orchestrator can flag
problematic source content for re-processing on the next pipeline run.

Feedback is stored as JSONL in state/quality_feedback/ and read by the
orchestrator before each planning phase.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default feedback directory
_STATE_DIR = Path(__file__).resolve().parent.parent / "state"
FEEDBACK_DIR = _STATE_DIR / "quality_feedback"


@dataclass
class QualityFeedback:
    """A quality feedback event from a downstream stage."""

    source_stage: str  # "trainforge", "libv2", "courseforge"
    target_stage: str  # "dart", "courseforge", "trainforge"
    course_code: str
    module_id: str  # Specific module/section that caused issues
    feedback_type: str  # "low_question_quality", "insufficient_content", etc.
    severity: str  # "critical", "high", "medium", "low"
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityFeedbackStore:
    """
    File-based store for quality feedback events.

    Each feedback event is appended to a JSONL file keyed by course code.
    The orchestrator reads pending feedback before planning phases and
    flags content for re-processing.
    """

    def __init__(self, feedback_dir: Optional[Path] = None):
        self.feedback_dir = Path(feedback_dir or FEEDBACK_DIR)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def log_feedback(self, feedback: QualityFeedback) -> None:
        """Append a quality feedback event."""
        path = self.feedback_dir / f"{feedback.course_code}.jsonl"
        line = json.dumps(feedback.to_dict()) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        logger.info(
            "Quality feedback logged: %s -> %s for %s/%s (%s)",
            feedback.source_stage,
            feedback.target_stage,
            feedback.course_code,
            feedback.module_id,
            feedback.feedback_type,
        )

    def get_pending_feedback(
        self,
        course_code: Optional[str] = None,
        target_stage: Optional[str] = None,
    ) -> List[QualityFeedback]:
        """
        Read pending feedback events, optionally filtered.

        Args:
            course_code: Filter to specific course
            target_stage: Filter to specific target stage

        Returns:
            List of QualityFeedback events
        """
        feedback_list = []
        pattern = f"{course_code}.jsonl" if course_code else "*.jsonl"

        for path in self.feedback_dir.glob(pattern):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        fb = QualityFeedback(**data)
                        if target_stage and fb.target_stage != target_stage:
                            continue
                        feedback_list.append(fb)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Invalid feedback line in %s: %s", path, e)

        return feedback_list

    def clear_feedback(self, course_code: str) -> int:
        """
        Clear feedback for a course (after it has been addressed).

        Returns number of events cleared.
        """
        path = self.feedback_dir / f"{course_code}.jsonl"
        if not path.exists():
            return 0

        count = sum(1 for line in open(path) if line.strip())
        path.unlink()
        logger.info("Cleared %d feedback events for %s", count, course_code)
        return count

    def get_flagged_modules(
        self,
        course_code: str,
        min_severity: str = "medium",
    ) -> List[str]:
        """
        Get module IDs flagged for re-processing.

        Args:
            course_code: Course to check
            min_severity: Minimum severity to include

        Returns:
            List of unique module IDs that need attention
        """
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_order = severity_order.get(min_severity, 1)

        feedback = self.get_pending_feedback(course_code=course_code)
        flagged = set()
        for fb in feedback:
            if severity_order.get(fb.severity, 0) >= min_order:
                flagged.add(fb.module_id)

        return sorted(flagged)


def log_quality_feedback(
    source_stage: str,
    target_stage: str,
    course_code: str,
    module_id: str,
    feedback_type: str,
    severity: str,
    message: str,
    **kwargs,
) -> None:
    """Convenience function to log quality feedback."""
    store = QualityFeedbackStore()
    store.log_feedback(
        QualityFeedback(
            source_stage=source_stage,
            target_stage=target_stage,
            course_code=course_code,
            module_id=module_id,
            feedback_type=feedback_type,
            severity=severity,
            message=message,
            **kwargs,
        )
    )
