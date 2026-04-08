"""
Training Data Analysis Tools

Provides tools for analyzing captured decision data quality and distribution.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import TRAINING_DIR

logger = logging.getLogger(__name__)

# Quality level ordering for comparison
QUALITY_ORDER = {"inadequate": 0, "developing": 1, "proficient": 2, "exemplary": 3}

# Training output paths
TRAINING_OUTPUT = TRAINING_DIR / "trainforge"
COURSEFORGE_OUTPUT = TRAINING_DIR / "courseforge"
DART_OUTPUT = TRAINING_DIR / "dart"


def _load_all_decision_records() -> List[Dict[str, Any]]:
    """Load all decision records from training captures."""
    records = []

    # Search all capture directories
    for output_dir in [TRAINING_OUTPUT, COURSEFORGE_OUTPUT, DART_OUTPUT]:
        if not output_dir.exists():
            continue

        for jsonl_file in output_dir.rglob("decisions_*.jsonl"):
            try:
                with open(jsonl_file) as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                record = json.loads(line)
                                record["_source_file"] = str(jsonl_file)
                                record["_line_num"] = line_num
                                records.append(record)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON at {jsonl_file}:{line_num}")
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")

    return records


def _compute_rationale_depth_score(rationale: str) -> float:
    """
    Score rationale quality beyond just length.

    Returns 0.0-1.0 composite score based on:
    - Length (0-0.3)
    - Reasoning indicators (0-0.4)
    - Pedagogical specificity (0-0.3)
    """
    if not rationale:
        return 0.0

    score = 0.0
    rationale_lower = rationale.lower()

    # Length component (0-0.3)
    score += min(0.3, len(rationale) / 300)

    # Reasoning indicators (0-0.4)
    reasoning_words = [
        "because", "since", "therefore", "allows", "enables",
        "prevents", "ensures", "supports", "aligns with", "chosen over",
        "preferred", "better than", "more appropriate", "rather than"
    ]
    reasoning_count = sum(1 for word in reasoning_words if word in rationale_lower)
    score += min(0.4, reasoning_count * 0.05)

    # Pedagogical specificity indicators (0-0.3)
    pedagogy_terms = [
        ("bloom", "taxonomy", "cognitive"),  # Bloom's taxonomy
        ("misconception", "distractor", "plausibility"),  # Assessment design
        ("learner", "student", "pedagog"),  # Learner-centered
        ("objective", "outcome", "competency"),  # Learning objectives
        ("udl", "accessibility", "inclusive"),  # UDL principles
    ]
    for term_group in pedagogy_terms:
        if any(term in rationale_lower for term in term_group):
            score += 0.06

    return min(1.0, score)


def _compute_composite_quality_score(record: Dict[str, Any]) -> float:
    """
    Compute 0.0-1.0 composite quality score for a decision record.

    Weights:
    - Rationale depth: 40%
    - Alternatives present: 20%
    - Inputs referenced: 15%
    - Confidence calibration: 10%
    - Outcome accepted: 15%
    """
    rationale = record.get("rationale", "")
    alternatives = record.get("alternatives_considered", [])
    inputs_ref = record.get("inputs_ref", [])
    confidence = record.get("confidence", 0.5)
    outcome = record.get("outcome") or {}

    scores = {
        "rationale_depth": _compute_rationale_depth_score(rationale),
        "alternatives_present": 1.0 if alternatives else 0.0,
        "inputs_referenced": 1.0 if inputs_ref else 0.0,
        "confidence_calibration": confidence if isinstance(confidence, (int, float)) and 0 <= confidence <= 1 else 0.5,
        "outcome_accepted": 1.0 if outcome.get("accepted") else 0.5,
    }

    weights = {
        "rationale_depth": 0.4,
        "alternatives_present": 0.2,
        "inputs_referenced": 0.15,
        "confidence_calibration": 0.1,
        "outcome_accepted": 0.15,
    }

    return sum(scores[k] * weights[k] for k in scores)


def _get_quality_level(record: Dict[str, Any]) -> str:
    """Extract quality level from record metadata."""
    metadata = record.get("metadata", {})
    return metadata.get("quality_level", "unknown")


def _content_hash(record: Dict[str, Any]) -> str:
    """Generate content hash for deduplication."""
    # Hash key fields that define uniqueness
    key_content = json.dumps({
        "decision_type": record.get("decision_type", ""),
        "decision": record.get("decision", ""),
        "rationale": record.get("rationale", "")[:100],
    }, sort_keys=True)
    return hashlib.md5(key_content.encode()).hexdigest()[:12]


def register_analysis_tools(mcp):
    """Register analysis tools with the MCP server."""

    @mcp.tool()
    async def analyze_training_data() -> str:
        """
        Analyze captured training data quality and distribution.

        Returns comprehensive statistics including:
        - Total records by tool and quality level
        - Rationale length statistics
        - Decision type distribution
        - Export readiness assessment
        - Recommendations for improvement
        """
        try:
            records = _load_all_decision_records()

            if not records:
                return json.dumps({
                    "total_records": 0,
                    "message": "No decision records found in training-captures/"
                })

            # Initialize counters
            by_tool = defaultdict(int)
            by_quality = defaultdict(int)
            by_decision_type = defaultdict(int)
            rationale_lengths = []
            depth_scores = []

            # Records with specific features
            with_alternatives = 0
            with_inputs_ref = 0
            with_outcome_accepted = 0

            # Content hashes for deduplication analysis
            content_hashes = set()
            duplicates = 0

            for record in records:
                # Tool distribution
                tool = record.get("tool", "unknown")
                by_tool[tool] += 1

                # Quality distribution
                quality = _get_quality_level(record)
                by_quality[quality] += 1

                # Decision type distribution
                decision_type = record.get("decision_type", "unknown")
                by_decision_type[decision_type] += 1

                # Rationale analysis
                rationale = record.get("rationale", "")
                rationale_lengths.append(len(rationale))
                depth_scores.append(_compute_rationale_depth_score(rationale))

                # Feature presence
                if record.get("alternatives_considered"):
                    with_alternatives += 1
                if record.get("inputs_ref"):
                    with_inputs_ref += 1
                if record.get("outcome", {}).get("accepted"):
                    with_outcome_accepted += 1

                # Deduplication check
                h = _content_hash(record)
                if h in content_hashes:
                    duplicates += 1
                else:
                    content_hashes.add(h)

            # Calculate statistics
            total = len(records)
            rationale_lengths.sort()
            depth_scores.sort()

            def percentile(data, p):
                if not data:
                    return 0
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (k - f) * (data[c] - data[f])

            # Count proficient+ for export readiness
            proficient_plus = by_quality.get("proficient", 0) + by_quality.get("exemplary", 0)

            # Build recommendations
            recommendations = []

            developing_pct = by_quality.get("developing", 0) / total * 100 if total > 0 else 0
            if developing_pct > 80:
                recommendations.append(
                    f"{developing_pct:.1f}% of data at 'developing' quality - consider improving rationale prompts"
                )

            if with_alternatives < total * 0.1:
                recommendations.append(
                    f"Only {with_alternatives} records ({with_alternatives/total*100:.1f}%) have alternatives_considered - add alternative tracking"
                )

            if by_tool.get("dart", 0) == 0:
                recommendations.append(
                    "No DART captures found - integrate decision capture into PDF conversion"
                )

            if duplicates > total * 0.05:
                recommendations.append(
                    f"{duplicates} duplicate records ({duplicates/total*100:.1f}%) detected - enable deduplication in export"
                )

            avg_depth = sum(depth_scores) / len(depth_scores) if depth_scores else 0
            if avg_depth < 0.4:
                recommendations.append(
                    f"Average rationale depth score is {avg_depth:.2f} - rationales need more substantive reasoning"
                )

            analysis = {
                "total_records": total,
                "analyzed_at": datetime.now().isoformat(),

                "by_tool": dict(by_tool),

                "by_quality": {
                    level: {
                        "count": count,
                        "pct": round(count / total * 100, 1) if total > 0 else 0
                    }
                    for level, count in sorted(by_quality.items(), key=lambda x: QUALITY_ORDER.get(x[0], -1))
                },

                "by_decision_type": {
                    dt: {
                        "count": count,
                        "pct": round(count / total * 100, 1) if total > 0 else 0
                    }
                    for dt, count in sorted(by_decision_type.items(), key=lambda x: -x[1])[:10]
                },

                "rationale_stats": {
                    "min_length": min(rationale_lengths) if rationale_lengths else 0,
                    "max_length": max(rationale_lengths) if rationale_lengths else 0,
                    "median_length": int(percentile(rationale_lengths, 50)),
                    "mean_length": round(sum(rationale_lengths) / len(rationale_lengths), 1) if rationale_lengths else 0,
                    "p25_length": int(percentile(rationale_lengths, 25)),
                    "p75_length": int(percentile(rationale_lengths, 75)),
                },

                "depth_score_stats": {
                    "min": round(min(depth_scores), 3) if depth_scores else 0,
                    "max": round(max(depth_scores), 3) if depth_scores else 0,
                    "mean": round(sum(depth_scores) / len(depth_scores), 3) if depth_scores else 0,
                    "median": round(percentile(depth_scores, 50), 3),
                },

                "feature_presence": {
                    "with_alternatives": with_alternatives,
                    "with_alternatives_pct": round(with_alternatives / total * 100, 1) if total > 0 else 0,
                    "with_inputs_ref": with_inputs_ref,
                    "with_inputs_ref_pct": round(with_inputs_ref / total * 100, 1) if total > 0 else 0,
                    "with_outcome_accepted": with_outcome_accepted,
                    "with_outcome_accepted_pct": round(with_outcome_accepted / total * 100, 1) if total > 0 else 0,
                },

                "deduplication": {
                    "unique_records": total - duplicates,
                    "duplicates_found": duplicates,
                    "duplicate_pct": round(duplicates / total * 100, 1) if total > 0 else 0,
                },

                "export_readiness": {
                    "proficient_plus_count": proficient_plus,
                    "proficient_plus_pct": round(proficient_plus / total * 100, 1) if total > 0 else 0,
                    "dpo_pairable_estimate": min(with_alternatives, proficient_plus // 2),
                    "ready_for_finetuning": proficient_plus >= 100,
                },

                "recommendations": recommendations,
            }

            return json.dumps(analysis, indent=2)

        except Exception as e:
            logger.exception("Error analyzing training data")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_quality_distribution(min_quality: str = "developing") -> str:
        """
        Get quality distribution with filtering preview.

        Args:
            min_quality: Minimum quality level to include ("inadequate", "developing", "proficient", "exemplary")

        Returns:
            Count of records at each quality level that would pass the filter
        """
        try:
            records = _load_all_decision_records()

            if not records:
                return json.dumps({"total": 0, "message": "No records found"})

            min_order = QUALITY_ORDER.get(min_quality, 1)

            distribution = defaultdict(int)
            passing = 0

            for record in records:
                quality = _get_quality_level(record)
                distribution[quality] += 1

                if QUALITY_ORDER.get(quality, 0) >= min_order:
                    passing += 1

            return json.dumps({
                "total_records": len(records),
                "min_quality_filter": min_quality,
                "passing_filter": passing,
                "passing_pct": round(passing / len(records) * 100, 1) if records else 0,
                "distribution": {
                    level: {
                        "count": distribution.get(level, 0),
                        "passes_filter": QUALITY_ORDER.get(level, 0) >= min_order
                    }
                    for level in ["inadequate", "developing", "proficient", "exemplary"]
                }
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def preview_export_filter(
        min_quality: str = "developing",
        min_confidence: float = 0.0,
        require_accepted: bool = False,
        deduplicate: bool = True
    ) -> str:
        """
        Preview how many records would be exported with given filters.

        Args:
            min_quality: Minimum quality level ("inadequate", "developing", "proficient", "exemplary")
            min_confidence: Minimum confidence score (0.0-1.0)
            require_accepted: Only include records with outcome.accepted=True
            deduplicate: Remove duplicate decisions based on content hash

        Returns:
            Filter statistics showing how many records pass each filter stage
        """
        try:
            records = _load_all_decision_records()

            if not records:
                return json.dumps({"total": 0, "message": "No records found"})

            min_order = QUALITY_ORDER.get(min_quality, 1)

            # Apply filters sequentially and track counts
            stages = {
                "total_scanned": len(records),
                "passed_quality": 0,
                "passed_confidence": 0,
                "passed_accepted": 0,
                "after_deduplication": 0,
            }

            filtered = records

            # Quality filter
            filtered = [
                r for r in filtered
                if QUALITY_ORDER.get(_get_quality_level(r), 0) >= min_order
            ]
            stages["passed_quality"] = len(filtered)

            # Confidence filter
            filtered = [
                r for r in filtered
                if r.get("confidence", 0.5) >= min_confidence
            ]
            stages["passed_confidence"] = len(filtered)

            # Accepted filter
            if require_accepted:
                filtered = [
                    r for r in filtered
                    if r.get("outcome", {}).get("accepted", False)
                ]
            stages["passed_accepted"] = len(filtered)

            # Deduplication
            if deduplicate:
                seen = set()
                deduped = []
                for r in filtered:
                    h = _content_hash(r)
                    if h not in seen:
                        seen.add(h)
                        deduped.append(r)
                filtered = deduped
            stages["after_deduplication"] = len(filtered)

            return json.dumps({
                "filters_applied": {
                    "min_quality": min_quality,
                    "min_confidence": min_confidence,
                    "require_accepted": require_accepted,
                    "deduplicate": deduplicate,
                },
                "filter_stages": stages,
                "final_count": len(filtered),
                "retention_rate": round(len(filtered) / len(records) * 100, 1) if records else 0,
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    logger.info("Analysis tools registered")
