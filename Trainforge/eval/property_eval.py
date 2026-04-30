"""Wave 109 — Phase C per-property evaluator.

Filters holdout probes by property surface forms (via the course's
property manifest), then runs the model on each filtered subset and
reports per-property accuracy. This is the eval-side companion to
``PropertyCoverageValidator`` (synthesis-side) — together they ensure
each declared property has training signal AND eval coverage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class PerPropertyEvaluator:
    """Score the model per-property using holdout probes filtered by
    surface-form match."""

    def __init__(
        self,
        holdout_split: Path,
        course_slug: str,
        model_callable: Callable[[str], str],
        max_questions_per_property: Optional[int] = None,
    ) -> None:
        self.holdout_path = Path(holdout_split)
        self.course_slug = course_slug
        self.model_callable = model_callable
        self.max_q = max_questions_per_property

    def evaluate(self) -> Dict[str, Any]:
        from Trainforge.eval.faithfulness import (
            _classify_response,
            _format_probe,
        )
        from lib.ontology.property_manifest import load_property_manifest

        try:
            manifest = load_property_manifest(self.course_slug)
        except FileNotFoundError:
            logger.info(
                "PerPropertyEvaluator: no property manifest for course "
                "'%s'; returning empty per-property report.",
                self.course_slug,
            )
            return {
                "per_property_accuracy": {},
                "per_property_scored": {},
                "course_slug": self.course_slug,
            }

        try:
            payload = json.loads(self.holdout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "PerPropertyEvaluator: cannot read holdout %s (%s)",
                self.holdout_path, exc,
            )
            return {
                "per_property_accuracy": {},
                "per_property_scored": {},
                "course_slug": self.course_slug,
            }

        # Audit 2026-04-30 fix: prefer the new `property_probes` array
        # emitted by HoldoutBuilder._build_property_probes. The legacy
        # `withheld_edges` filter remains for backward compat with
        # holdout splits built before the fix landed.
        property_probes = payload.get("property_probes") or []
        edges = payload.get("withheld_edges", []) or []

        per_property_accuracy: Dict[str, Optional[float]] = {}
        per_property_scored: Dict[str, int] = {}

        for prop in manifest.properties:
            matching = []

            # Source 1: explicit property_probes (fast path; the probe
            # is already stamped with `property_id`).
            for probe in property_probes:
                if probe.get("property_id") == prop.id:
                    matching.append((probe, probe.get("probe_text") or probe.get("prompt", "")))

            # Source 2: legacy edges-based filter (back-compat path).
            if not matching:
                for edge in edges:
                    probe_text = edge.get("probe_text") or _format_probe(edge)
                    source = str(edge.get("source") or "")
                    target = str(edge.get("target") or "")
                    haystack = f"{probe_text} {source} {target}"
                    if prop.matches(haystack):
                        matching.append((edge, probe_text))

            if self.max_q is not None:
                matching = matching[: self.max_q]

            if not matching:
                per_property_accuracy[prop.id] = None
                per_property_scored[prop.id] = 0
                continue

            correct = 0
            for _edge, probe_text in matching:
                try:
                    response = self.model_callable(probe_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "model_callable raised for %s: %s", prop.id, exc
                    )
                    continue
                if _classify_response(str(response)) == "affirm":
                    correct += 1
            per_property_accuracy[prop.id] = correct / len(matching)
            per_property_scored[prop.id] = len(matching)

        return {
            "per_property_accuracy": per_property_accuracy,
            "per_property_scored": per_property_scored,
            "course_slug": self.course_slug,
        }


__all__ = ["PerPropertyEvaluator"]
