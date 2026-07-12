"""Course-intelligence-level top-level aggregator (Worker W4.6).

A deterministic (no-model) 0-5 self-assessment of a built course's
capability surface. Tallies PRESENT capability artifacts across five
axes, one point each:

1. ``key_terms_glossary`` — present when the concept graph carries at
   least one glossary-definition edge (``is-a`` from
   ``is_a_from_key_terms`` OR ``defined-by`` from
   ``defined_by_from_first_mention``): the key-terms glossary surface.
2. ``concept_graph`` — present when
   ``concept_graph_semantic.json`` exists with >=1 node AND >=1 edge.
3. ``assessment_density`` — present when the assessment-item density
   (``(questions + self_check_chunks) / max(1, objectives)``) clears
   ``_ASSESSMENT_DENSITY_FLOOR`` (0.5 — at least one assessment item per
   two objectives on average).
4. ``prereq_cross_links`` — present when the concept graph carries at
   least one ``prerequisite`` edge (concept-level prerequisite
   scaffolding). Content-derived prerequisite edges (W3.1
   ``prerequisite_from_definition_mention``) are surfaced as bonus
   evidence.
5. ``faq`` — the FAQ capability is NOT built yet, so this axis is
   scored ABSENT unconditionally (documented gap; reserved so the
   rubric size stays fixed at 5 when FAQ lands).

Output: ``<libv2_course>/intelligence_level_report.json``.

Opt-in posture (W4.6): the aggregator only runs when
``ED4ALL_INTELLIGENCE_RUBRIC`` is truthy — the ``WorkflowRunner``
post-loop helper short-circuits on the OFF (default) resolver, so a
default run is byte-identical (no file emitted).
:func:`resolve_intelligence_rubric` mirrors the parse-with-fallback
contract of the other opt-in flags.

Best-effort posture (matches ``CoverageMapAggregator``): missing inputs
degrade gracefully — a course with no concept graph / assessments simply
scores those axes absent rather than crashing, and ``WorkflowRunner``
wraps the call in try/except so an aggregator failure does NOT change
``final_status``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

ENV_INTELLIGENCE_RUBRIC = "ED4ALL_INTELLIGENCE_RUBRIC"
_TRUTHY = {"1", "true", "yes", "on"}

#: Minimum (questions + self_check_chunks) / objectives ratio for the
#: assessment_density axis to score present.
_ASSESSMENT_DENSITY_FLOOR = 0.5

#: Chunk-type labels counted as self-check / assessment content.
_ASSESSMENT_CHUNK_TYPES = {
    "self_check",
    "self_check_question",
    "assessment",
    "assessment_item",
    "question",
    "quiz",
}

#: Canonical axis order (also the rubric size).
_AXIS_ORDER = [
    "key_terms_glossary",
    "concept_graph",
    "assessment_density",
    "prereq_cross_links",
    "faq",
]
MAX_SCORE = 5


def resolve_intelligence_rubric(value: object = None) -> bool:
    """Return True iff the intelligence-rubric aggregation is enabled.

    ``value`` overrides the env when not ``None``. Falsey / garbage /
    unset -> False (parse-with-fallback, mirroring the other opt-in
    ``ED4ALL_*`` toggles).
    """
    if value is None:
        raw = os.environ.get(ENV_INTELLIGENCE_RUBRIC, "")
    else:
        raw = str(value)
    return raw.strip().lower() in _TRUTHY


class IntelligenceLevelAggregator:
    """Score a course's capability surface on a deterministic 0-5 rubric.

    Parameters
    ----------
    course_code / run_id:
        Operator-facing course code + workflow ID (surfaced verbatim).
    libv2_course_path:
        Optional override for the LibV2 course directory
        (``LibV2/courses/<slug>``). Root for the concept graph, chunks,
        objectives, and assessments probes.
    trainforge_dir:
        Optional fallback root for chunks / objectives / assessments
        when archival hasn't run.
    concept_graph_path:
        Optional explicit concept-graph path (wins over the course-dir
        probe).
    decision_capture:
        Optional ``DecisionCapture`` instance. When wired the aggregator
        emits one ``intelligence_level_scored`` event per ``build()``.
    """

    def __init__(
        self,
        *,
        course_code: str = "",
        run_id: str = "",
        libv2_course_path: Optional[Path] = None,
        trainforge_dir: Optional[Path] = None,
        concept_graph_path: Optional[Path] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        self.libv2_course_path = (
            Path(libv2_course_path) if libv2_course_path else None
        )
        self.trainforge_dir = (
            Path(trainforge_dir) if trainforge_dir else None
        )
        self.concept_graph_path = (
            Path(concept_graph_path) if concept_graph_path else None
        )
        self.decision_capture = decision_capture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Score every axis and build the intelligence-level report dict."""
        graph = self._load_concept_graph()
        nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
        edges = graph.get("edges") if isinstance(graph, Mapping) else None
        nodes = nodes if isinstance(nodes, list) else []
        edges = edges if isinstance(edges, list) else []

        # Edge-type tallies for the graph-derived axes.
        glossary_is_a = 0
        glossary_defined_by = 0
        prereq_edges = 0
        content_prereq_edges = 0
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            etype = str(edge.get("type") or "").strip()
            if etype == "is-a":
                glossary_is_a += 1
            elif etype == "defined-by":
                glossary_defined_by += 1
            elif etype == "prerequisite":
                prereq_edges += 1
                if self._edge_rule(edge) == "prerequisite_from_definition_mention":
                    content_prereq_edges += 1

        # Assessment-density inputs.
        questions_count = self._count_questions()
        self_check_chunks = self._count_self_check_chunks()
        objectives_count = self._count_objectives()
        assessment_items = questions_count + self_check_chunks
        density = assessment_items / max(1, objectives_count)

        # ------------------------------------------------------------------
        # Axis verdicts.
        # ------------------------------------------------------------------
        axes: List[Dict[str, Any]] = []

        glossary_present = (glossary_is_a + glossary_defined_by) > 0
        axes.append({
            "axis": "key_terms_glossary",
            "present": glossary_present,
            "evidence": {
                "is_a_edges": glossary_is_a,
                "defined_by_edges": glossary_defined_by,
            },
        })

        graph_present = bool(nodes) and bool(edges)
        axes.append({
            "axis": "concept_graph",
            "present": graph_present,
            "evidence": {
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        })

        density_present = density >= _ASSESSMENT_DENSITY_FLOOR
        axes.append({
            "axis": "assessment_density",
            "present": density_present,
            "evidence": {
                "questions": questions_count,
                "self_check_chunks": self_check_chunks,
                "assessment_items": assessment_items,
                "objectives": objectives_count,
                "density": round(density, 4),
                "density_floor": _ASSESSMENT_DENSITY_FLOOR,
            },
        })

        prereq_present = prereq_edges > 0
        axes.append({
            "axis": "prereq_cross_links",
            "present": prereq_present,
            "evidence": {
                "prerequisite_edges": prereq_edges,
                "content_derived_prerequisite_edges": content_prereq_edges,
            },
        })

        # FAQ capability is not built yet — scored absent unconditionally.
        axes.append({
            "axis": "faq",
            "present": False,
            "evidence": {
                "note": "FAQ capability not built yet; axis reserved.",
            },
        })

        present_axes = [a["axis"] for a in axes if a["present"]]
        absent_axes = [a["axis"] for a in axes if not a["present"]]
        score = len(present_axes)

        report = {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "score": score,
            "max_score": MAX_SCORE,
            "axes": axes,
            "summary": {
                "score": score,
                "max_score": MAX_SCORE,
                "present_axes": present_axes,
                "absent_axes": absent_axes,
            },
        }

        self._emit_aggregator_decision(report)
        return report

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic).

        Returns the resolved path on success. Raises ``OSError`` on
        filesystem failure — the workflow-runner caller wraps the call in
        try/except (best-effort; the per-phase reports remain the source
        of truth).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build()
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    # Helpers — input resolution
    # ------------------------------------------------------------------
    def _load_concept_graph(self) -> Dict[str, Any]:
        """Best-effort concept-graph load (both LibV2 layouts)."""
        if self.concept_graph_path is not None:
            payload = self._read_json(self.concept_graph_path)
            if isinstance(payload, dict):
                return payload
        if self.libv2_course_path is not None:
            for subdir in ("graph", "concept_graph"):
                cand = (
                    self.libv2_course_path
                    / subdir
                    / "concept_graph_semantic.json"
                )
                if cand.exists():
                    payload = self._read_json(cand)
                    if isinstance(payload, dict):
                        return payload
        return {}

    def _count_questions(self) -> int:
        """Best-effort count of assessment questions across assessments.json."""
        candidates: List[Path] = []
        if self.trainforge_dir is not None:
            candidates.append(self.trainforge_dir / "assessments.json")
            candidates.append(
                self.trainforge_dir / "training_specs" / "assessments.json"
            )
        if self.libv2_course_path is not None:
            candidates.append(
                self.libv2_course_path
                / "training_specs"
                / "assessments.json"
            )
            candidates.append(self.libv2_course_path / "assessments.json")
        for cand in candidates:
            if cand.exists():
                payload = self._read_json(cand)
                if isinstance(payload, Mapping):
                    questions = payload.get("questions")
                    if isinstance(questions, list):
                        return sum(
                            1 for q in questions if isinstance(q, Mapping)
                        )
        return 0

    def _count_self_check_chunks(self) -> int:
        """Best-effort count of self-check / assessment chunks."""
        return sum(
            1
            for chunk in self._iter_chunks()
            if str(chunk.get("chunk_type") or "").strip().lower()
            in _ASSESSMENT_CHUNK_TYPES
        )

    def _iter_chunks(self) -> List[Dict[str, Any]]:
        """Best-effort load of the imscc/dart chunkset (first that resolves)."""
        candidates: List[Path] = []
        if self.libv2_course_path is not None:
            candidates.append(
                self.libv2_course_path / "imscc_chunks" / "chunks.jsonl"
            )
            # DART->semantik purge Stage 3c: the ratified staged emit dir is
            # ``semantik_chunks/``; try it before the legacy ``dart_chunks/``.
            candidates.append(
                self.libv2_course_path / "semantik_chunks" / "chunks.jsonl"
            )
            candidates.append(
                self.libv2_course_path / "dart_chunks" / "chunks.jsonl"
            )
        if self.trainforge_dir is not None:
            candidates.append(
                self.trainforge_dir / "imscc_chunks" / "chunks.jsonl"
            )
            candidates.append(self.trainforge_dir / "chunks.jsonl")
        for cand in candidates:
            if cand.exists():
                rows = self._read_jsonl(cand)
                if rows:
                    return rows
        return []

    def _count_objectives(self) -> int:
        """Best-effort count of synthesized objectives (denominator)."""
        from lib.validators.abcd_objective import _flatten_objectives

        candidates: List[Path] = []
        if self.libv2_course_path is not None:
            for fname in ("objectives.json", "synthesized_objectives.json"):
                candidates.append(self.libv2_course_path / fname)
        if self.trainforge_dir is not None:
            for fname in (
                "objectives.json",
                "synthesized_objectives.json",
                "training_specs/objectives.json",
            ):
                candidates.append(self.trainforge_dir / fname)
        for cand in candidates:
            if cand.exists():
                payload = self._read_json(cand)
                if payload is not None:
                    flat = _flatten_objectives(payload)
                    if flat:
                        return len(flat)
        return 0

    @staticmethod
    def _edge_rule(edge: Mapping[str, Any]) -> str:
        """Return the provenance rule name for an edge (empty if absent)."""
        prov = edge.get("provenance")
        if isinstance(prov, Mapping):
            raw = prov.get("rule")
            if isinstance(raw, str):
                return raw.strip()
        return ""

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        """Best-effort JSON read; returns ``None`` on any failure."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "intelligence_level: cannot read %s: %s", path, exc,
            )
            return None

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        """Best-effort JSONL read; skips malformed rows."""
        out: List[Dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "intelligence_level: cannot read %s: %s", path, exc,
            )
            return out
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------
    def _emit_aggregator_decision(self, report: Mapping[str, Any]) -> None:
        """Emit one ``intelligence_level_scored`` decision per build.

        Best-effort: any failure is logged and swallowed. Rationale
        interpolates the score + per-axis verdicts so it clears the
        20-char floor and stays replayable post-hoc.
        """
        capture = self.decision_capture
        if capture is None:
            return
        try:
            summary = report.get("summary") or {}
            score = summary.get("score", 0)
            present = summary.get("present_axes") or []
            absent = summary.get("absent_axes") or []
            rationale = (
                f"Intelligence-level scored: score={score}/{MAX_SCORE}, "
                f"present={sorted(present)}, absent={sorted(absent)}"
            )
            capture.log_decision(
                decision_type="intelligence_level_scored",
                decision=f"score={score}/{MAX_SCORE}",
                rationale=rationale,
                context=str({
                    "course_code": report.get("course_code"),
                    "run_id": report.get("run_id"),
                    "schema_version": report.get("schema_version"),
                }),
                metrics={
                    "score": score,
                    "max_score": MAX_SCORE,
                    "present_axis_count": len(present),
                    "absent_axis_count": len(absent),
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "intelligence_level_scored: %s",
                exc,
            )


__all__ = [
    "IntelligenceLevelAggregator",
    "SCHEMA_VERSION",
    "ENV_INTELLIGENCE_RUBRIC",
    "MAX_SCORE",
    "resolve_intelligence_rubric",
]
