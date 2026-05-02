"""Wave 138a Phase A — Tier-2 graph-derived teaching-role alignment evaluator.

Reads a course's ``chunks.jsonl`` line by line and aggregates the
``teaching_role`` distribution per ``content_type_label``. Compares the
observed distribution against an expected-mode floor table and flags
mismatches.

The evaluator is **purely declarative**: no LLM dispatch, no graph
traversal, no model_callable. Wall-time on a 1000-chunk corpus is well
under 100ms.

Designed to expose the audit-confirmed gap where
``content_type_label="real_world_scenario"`` chunks systematically miss
the ``transfer`` teaching role because ``align_chunks._heuristic_role``
ignores ``content_type_label`` entirely. The evaluator gives the
operator empirical signal *before* a fix is selected.

Known trade-off: ``DEFAULT_MIN_CHUNKS_FOR_FLAG=5`` skips content types
with fewer chunks than the threshold, since the share statistic is
meaningless on N=1 / N=2. A corpus whose entire ``assessment`` block
is 4 mislabeled chunks will silently skip the assessment alignment
check; calibrate the threshold against multiple corpora before
promoting any downstream gate to ``critical``.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TeachingRoleAlignmentEvaluator:
    """Tier-2 graph-derived evaluator: aggregates teaching_role
    distribution per content_type_label and flags expected-mode
    mismatches.

    Reads chunks.jsonl directly — no LLM dispatch, no model_callable.
    Runs in seconds; safe to invoke unconditionally.
    """

    # Default expected-mode floors. Each entry maps a
    # ``content_type_label`` to ``{"expected_role", "min_share"}``.
    # Operators override via the ``expected_modes`` constructor kwarg
    # or via the workflow gate's ``inputs.expected_modes`` block. The
    # ``procedure`` and ``example`` content types are intentionally
    # absent: per the plan's Decision Points Q2/Q3, their pedagogical
    # role is bimodal in observed corpora and the operator has not yet
    # committed a default.
    DEFAULT_EXPECTED_MODES: Dict[str, Dict[str, Any]] = {
        "real_world_scenario": {"expected_role": "transfer",   "min_share": 0.70},
        "scenario":            {"expected_role": "transfer",   "min_share": 0.70},
        "definition":          {"expected_role": "introduce",  "min_share": 0.70},
        "summary":             {"expected_role": "synthesize", "min_share": 0.70},
        "assessment_item":     {"expected_role": "assess",     "min_share": 1.00},
        "assessment":          {"expected_role": "assess",     "min_share": 1.00},
        "self_check":          {"expected_role": "assess",     "min_share": 1.00},
    }

    # Skip content types with fewer than this many chunks — the
    # share statistic is meaningless on N=1 / N=2.
    DEFAULT_MIN_CHUNKS_FOR_FLAG: int = 5

    def __init__(
        self,
        chunks_path: Path,
        *,
        expected_modes: Optional[Dict[str, Dict[str, Any]]] = None,
        min_chunks_for_flag: int = DEFAULT_MIN_CHUNKS_FOR_FLAG,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        # Operator-supplied modes are merged OVER the defaults so a
        # caller can extend the table without losing the canonical
        # entries (e.g. add a ``procedure`` rule once the operator
        # pins one).
        merged: Dict[str, Dict[str, Any]] = dict(self.DEFAULT_EXPECTED_MODES)
        if expected_modes:
            for key, value in expected_modes.items():
                merged[key] = dict(value)
        self.expected_modes = merged
        self.min_chunks_for_flag = int(min_chunks_for_flag)

    def evaluate(self) -> Dict[str, Any]:
        """Group chunks by ``content_type_label``, count
        ``teaching_role`` occurrences, and compare against the
        expected-mode table.

        Returns a dict matching the frozen plan shape — see the module
        docstring + ``plans/eval-driven-teaching-role-alignment/plan.md``
        for the canonical schema.
        """
        # Group teaching_role values per content_type_label. Chunks
        # missing content_type_label are skipped — there's nothing to
        # group them by, and flagging an "unlabeled" pseudo-bucket
        # would inflate the summary's total_content_types.
        per_label_roles: Dict[str, Counter] = defaultdict(Counter)

        try:
            with self.chunks_path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "TeachingRoleAlignmentEvaluator: skipping "
                            "malformed JSON at %s:%d (%s)",
                            self.chunks_path, line_no, exc,
                        )
                        continue
                    label = chunk.get("content_type_label")
                    if label is None:
                        continue
                    role = chunk.get("teaching_role")
                    # Track all chunks with a label, even when their
                    # role is missing — the share calc divides by
                    # total chunks observed.
                    per_label_roles[str(label)][role if role is not None else "__missing__"] += 1
        except FileNotFoundError:
            logger.info(
                "TeachingRoleAlignmentEvaluator: chunks file %s not "
                "found; returning empty report.",
                self.chunks_path,
            )
            per_label_roles = defaultdict(Counter)

        content_type_role_alignment: Dict[str, Dict[str, Any]] = {}
        mismatched_content_types = []
        passing_with_rule = 0
        total_with_rule = 0

        for label in sorted(per_label_roles.keys()):
            counter = per_label_roles[label]
            total_chunks = sum(counter.values())
            # Materialize the role distribution dict in
            # most-common-first order. We keep ``__missing__`` out of
            # the surfaced distribution so an operator reading the
            # report doesn't see a synthetic key, but it still counts
            # toward ``total_chunks`` (a chunk with no role IS a
            # mislabeled chunk for alignment purposes).
            visible_dist: Dict[str, int] = {}
            for role, count in counter.most_common():
                if role == "__missing__":
                    continue
                visible_dist[role] = count

            # Dominant role: pick the most common visible role; ties
            # broken alphabetically for determinism. If every role is
            # ``__missing__``, dominant_role is None.
            if visible_dist:
                top_count = max(visible_dist.values())
                tied = sorted(
                    role for role, count in visible_dist.items()
                    if count == top_count
                )
                dominant_role: Optional[str] = tied[0]
            else:
                dominant_role = None

            rule = self.expected_modes.get(label)
            if rule is None:
                content_type_role_alignment[label] = {
                    "total_chunks": total_chunks,
                    "role_distribution": visible_dist,
                    "dominant_role": dominant_role,
                    "expected_role": None,
                    "expected_share": None,
                    "actual_expected_share": None,
                    "mismatch": None,
                    "skipped_below_threshold": False,
                }
                continue

            expected_role = rule["expected_role"]
            expected_share = float(rule["min_share"])
            expected_count = counter.get(expected_role, 0)
            actual_expected_share = (
                expected_count / total_chunks if total_chunks else 0.0
            )
            skipped = total_chunks < self.min_chunks_for_flag

            if skipped:
                mismatch: Optional[bool] = False
            else:
                mismatch = actual_expected_share < expected_share

            content_type_role_alignment[label] = {
                "total_chunks": total_chunks,
                "role_distribution": visible_dist,
                "dominant_role": dominant_role,
                "expected_role": expected_role,
                "expected_share": expected_share,
                "actual_expected_share": actual_expected_share,
                "mismatch": mismatch,
                "skipped_below_threshold": skipped,
            }

            total_with_rule += 1
            if mismatch is True:
                mismatched_content_types.append(label)
            else:
                # mismatch is False (either truly aligned OR skipped
                # below threshold). Skipped entries don't fail the
                # corpus-level alignment_rate — they're just
                # statistically uninformative. Counting them as
                # passing matches the plan's
                # "passing_with_rule / total_with_rule" definition.
                passing_with_rule += 1

        if total_with_rule > 0:
            alignment_rate = passing_with_rule / total_with_rule
        else:
            # Vacuously true: no rule means nothing can mismatch.
            alignment_rate = 1.0

        summary = {
            "total_content_types": len(content_type_role_alignment),
            "content_types_with_expected_mode": total_with_rule,
            "mismatched_content_types": mismatched_content_types,
            "alignment_rate": alignment_rate,
        }

        return {
            "content_type_role_alignment": content_type_role_alignment,
            "summary": summary,
        }


__all__ = ["TeachingRoleAlignmentEvaluator"]
