"""Coverage-map top-level aggregator (Worker W3.E).

Builds an objective-keyed coverage table linking objectives -> chunks ->
questions -> training_pairs across the full pipeline. Surfaces three
orphan classes to operators in one JSON:

* ``orphan_objectives`` — objectives that have no chunks AND no
  questions AND no training_pairs (a learning outcome the corpus
  doesn't actually deliver against).
* ``orphan_chunks`` — chunks whose ``learning_outcome_refs[]`` is
  empty OR cites no extant objective (chunks that exist in the
  corpus but never roll up to a published objective; downstream
  retrieval / training surfaces should ignore them).
* ``orphan_questions`` — assessment questions whose
  ``objective_id`` / ``objective_ids[]`` is empty OR cites no
  extant objective (questions that exist but don't probe a
  declared objective; the assessment generator's coverage gate
  should already block these but the aggregator surfaces the
  residue).

Output: ``<libv2_course>/coverage_map.json`` (or
``<trainforge_dir>/coverage_map.json`` when archival hasn't run).

Schema sketch (mirrors plan §"Worker W3.E")::

    {
      "schema_version": "1.0",
      "course_code": "<slug>",
      "run_id": "<workflow_id>",
      "generated_at": "<iso8601>",
      "objectives": [
        {
          "objective_id": "TO-01",
          "statement": "...",
          "chunks": ["<chunk_id>", ...],
          "questions": ["<question_id>", ...],
          "answers_grounded_in_chunks": ["<question_id>", ...],
          "training_pairs": ["<pair_id>", ...]
        }
      ],
      "summary": {
        "total_objectives": <int>,
        "objectives_with_chunks": <int>,
        "objectives_with_questions": <int>,
        "objectives_with_training_pairs": <int>,
        "orphan_objectives": ["<id>", ...],
        "orphan_chunks": ["<id>", ...],
        "orphan_questions": ["<id>", ...]
      }
    }

Best-effort posture: missing inputs degrade gracefully — the
aggregator emits a partial coverage_map with empty arrays rather
than crashing. ``WorkflowRunner`` wraps the aggregator call in
try/except so a build failure does NOT change ``final_status``;
the per-phase reports remain the source of truth.

Out-of-scope (Wave 4 candidates):

* JSONL emit option for >10k-row courses.
* Per-objective cosine similarity scoring of question stems
  against objective text. The chunk-set intersection between a
  question's ``source_chunks[]`` and the objective's collated
  chunk list is the day-1 grounding signal exposed via
  ``answers_grounded_in_chunks[]``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"


class CoverageMapAggregator:
    """Aggregate objective <-> chunk <-> question <-> training_pair links.

    Parameters
    ----------
    phase_outputs:
        ``WorkflowRunner.run_workflow``'s accumulated ``phase_outputs``
        map. The aggregator reads (best-effort) the
        ``course_planning`` / ``training_synthesis`` /
        ``trainforge_assessment`` / ``libv2_archival`` payloads
        without modifying them.
    course_code:
        Operator-facing course code. Surfaces as ``course_code`` so a
        diff across runs keys cleanly.
    run_id:
        Workflow ID.
    libv2_course_path:
        Optional override for the LibV2 course directory
        (``LibV2/courses/<slug>``). When unset the aggregator
        resolves this lazily from
        ``phase_outputs.libv2_archival.course_dir``.
    trainforge_dir:
        Optional override for the Trainforge workspace directory
        (used as the input + fallback emit root). When unset the
        aggregator resolves from
        ``phase_outputs.trainforge_assessment.trainforge_dir`` /
        ``phase_outputs.training_synthesis.corpus_dir``.
    objectives_path:
        Optional explicit path to a ``synthesized_objectives.json`` /
        ``objectives.json`` file. When unset the aggregator resolves
        in priority order:

        1. ``phase_outputs.course_planning.synthesized_objectives_path``
        2. ``<libv2_course>/objectives.json``
        3. ``<trainforge_dir>/objectives.json``
    decision_capture:
        Optional ``DecisionCapture`` instance. When wired the
        aggregator emits one ``coverage_map_aggregated`` event per
        ``build()`` call so the audit trail records the resolved
        per-bucket counts.
    """

    def __init__(
        self,
        phase_outputs: Mapping[str, Mapping[str, Any]],
        *,
        course_code: str = "",
        run_id: str = "",
        libv2_course_path: Optional[Path] = None,
        trainforge_dir: Optional[Path] = None,
        objectives_path: Optional[Path] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.phase_outputs = phase_outputs or {}
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        self.libv2_course_path = (
            Path(libv2_course_path) if libv2_course_path else None
        )
        self.trainforge_dir = (
            Path(trainforge_dir) if trainforge_dir else None
        )
        self.objectives_path = (
            Path(objectives_path) if objectives_path else None
        )
        self.decision_capture = decision_capture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Walk every input source and build the unified coverage_map dict."""
        # 1. Build the canonical objectives table from the synthesized /
        # archived objectives JSON. Empty list when no source resolvable
        # (graceful degradation — every downstream walk is no-op).
        objectives_flat = self._load_objectives()
        objectives_by_id: Dict[str, Dict[str, Any]] = {}
        for lo in objectives_flat:
            obj_id = self._objective_id(lo)
            if not obj_id:
                continue
            # Last writer wins on duplicate IDs (defensive — the
            # course-planner emits unique IDs by construction; a
            # malformed corpus may carry dupes).
            objectives_by_id[obj_id] = {
                "objective_id": obj_id,
                "statement": str(
                    lo.get("statement")
                    or lo.get("text")
                    or lo.get("description")
                    or ""
                ),
                "chunks": [],
                "questions": [],
                "answers_grounded_in_chunks": [],
                "training_pairs": [],
            }

        objective_ids: Set[str] = set(objectives_by_id.keys())

        # 2. Walk chunks. For each chunk, append its chunk_id to every
        # cited objective's chunks[] list. Track orphan chunks (empty
        # learning_outcome_refs[] OR all refs missing from the
        # objectives_by_id table).
        chunks = self._load_chunks()
        chunk_objective_map: Dict[str, Set[str]] = {}
        orphan_chunks: List[str] = []
        for chunk in chunks:
            chunk_id = self._chunk_id(chunk)
            if not chunk_id:
                continue
            refs_raw = chunk.get("learning_outcome_refs") or []
            refs = self._normalize_refs(refs_raw)
            covered_refs = [r for r in refs if r in objective_ids]
            if not refs or not covered_refs:
                orphan_chunks.append(chunk_id)
            chunk_objective_map[chunk_id] = set(covered_refs)
            for ref in covered_refs:
                lst = objectives_by_id[ref]["chunks"]
                if chunk_id not in lst:
                    lst.append(chunk_id)

        # 3. Walk questions (when an assessments.json was located).
        # For each question, append its question_id to the cited
        # objective's questions[] list AND to
        # answers_grounded_in_chunks[] when source_chunks[] is non-empty
        # AND those chunks intersect the objective's chunks[] list.
        questions = self._load_questions()
        orphan_questions: List[str] = []
        for question in questions:
            qid = self._question_id(question)
            if not qid:
                continue
            q_obj_refs = self._question_objective_ids(question)
            covered_q_refs = [r for r in q_obj_refs if r in objective_ids]
            if not q_obj_refs or not covered_q_refs:
                orphan_questions.append(qid)
                continue
            q_source_chunks = self._question_source_chunks(question)
            for ref in covered_q_refs:
                obj_entry = objectives_by_id[ref]
                if qid not in obj_entry["questions"]:
                    obj_entry["questions"].append(qid)
                if not q_source_chunks:
                    continue
                obj_chunk_set = set(obj_entry["chunks"])
                if obj_chunk_set & set(q_source_chunks):
                    if qid not in obj_entry["answers_grounded_in_chunks"]:
                        obj_entry["answers_grounded_in_chunks"].append(qid)

        # 4. Walk instruction_pairs.jsonl + preference_pairs.jsonl.
        # For each pair, append pair_id to lo_refs[]-cited objectives'
        # training_pairs[] lists.
        for pair in self._load_pairs():
            pair_id = self._pair_id(pair)
            if not pair_id:
                continue
            pair_refs = self._normalize_refs(pair.get("lo_refs") or [])
            for ref in pair_refs:
                if ref not in objective_ids:
                    continue
                lst = objectives_by_id[ref]["training_pairs"]
                if pair_id not in lst:
                    lst.append(pair_id)

        # 5. Build summary + emit.
        objectives_list = sorted(
            objectives_by_id.values(),
            key=lambda obj: obj["objective_id"],
        )
        summary = self._build_summary(
            objectives_list,
            orphan_chunks=orphan_chunks,
            orphan_questions=orphan_questions,
        )

        report = {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "objectives": objectives_list,
            "summary": summary,
        }

        # Best-effort decision-capture emit; never raises into the
        # aggregator build path.
        self._emit_aggregator_decision(report)

        return report

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic).

        Returns the resolved absolute path on success. Raises ``OSError``
        on filesystem failure — the workflow-runner caller wraps the
        call in try/except so an aggregator failure does not abort the
        workflow (aggregation is best-effort; the per-phase reports
        remain the source of truth).
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
    def _resolve_libv2_course_path(self) -> Optional[Path]:
        """Resolve the LibV2 course dir from explicit override or phase output."""
        if self.libv2_course_path is not None:
            return self.libv2_course_path
        archival = self.phase_outputs.get("libv2_archival") or {}
        course_dir = archival.get("course_dir")
        if course_dir:
            return Path(course_dir)
        return None

    def _resolve_trainforge_dir(self) -> Optional[Path]:
        """Resolve the Trainforge workspace dir from explicit override / phase outputs."""
        if self.trainforge_dir is not None:
            return self.trainforge_dir
        ta = self.phase_outputs.get("trainforge_assessment") or {}
        tdir = ta.get("trainforge_dir")
        if tdir:
            return Path(tdir)
        ts = self.phase_outputs.get("training_synthesis") or {}
        cdir = ts.get("corpus_dir") or ts.get("trainforge_dir")
        if cdir:
            return Path(cdir)
        return None

    def _load_objectives(self) -> List[Dict[str, Any]]:
        """Best-effort load of the objectives table.

        Priority chain:

        1. Explicit ``self.objectives_path`` override.
        2. ``phase_outputs.course_planning.synthesized_objectives_path``.
        3. ``<libv2_course>/objectives.json``
           (LibV2-archive form, post-archival).
        4. ``<trainforge_dir>/objectives.json``
           (some pipelines emit this here too).

        Returns ``[]`` when no source resolvable — every downstream
        walk no-ops on an empty objectives map.
        """
        # ``_flatten_objectives`` handles both the Courseforge synthesized
        # form (terminal_objectives + chapter_objectives) AND the LibV2
        # archive form (terminal_outcomes + component_objectives), so the
        # caller doesn't need to know which shape is on disk.
        from lib.validators.abcd_objective import _flatten_objectives

        # Path 1 — explicit override.
        if self.objectives_path is not None:
            payload = self._read_json(self.objectives_path)
            if payload is not None:
                return _flatten_objectives(payload)

        # Path 2 — phase_outputs.course_planning.synthesized_objectives_path.
        cp = self.phase_outputs.get("course_planning") or {}
        cp_path = cp.get("synthesized_objectives_path")
        if cp_path:
            payload = self._read_json(Path(cp_path))
            if payload is not None:
                return _flatten_objectives(payload)

        # Path 3 — LibV2-archive form.
        course_path = self._resolve_libv2_course_path()
        if course_path is not None:
            for fname in ("objectives.json", "synthesized_objectives.json"):
                cand = course_path / fname
                if cand.exists():
                    payload = self._read_json(cand)
                    if payload is not None:
                        return _flatten_objectives(payload)

        # Path 4 — Trainforge dir fallback.
        tdir = self._resolve_trainforge_dir()
        if tdir is not None:
            for fname in (
                "objectives.json",
                "synthesized_objectives.json",
                "training_specs/objectives.json",
            ):
                cand = tdir / fname
                if cand.exists():
                    payload = self._read_json(cand)
                    if payload is not None:
                        return _flatten_objectives(payload)

        return []

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Best-effort load of ``imscc_chunks/chunks.jsonl``.

        Priority chain:

        1. ``<libv2_course>/imscc_chunks/chunks.jsonl``.
        2. ``<trainforge_dir>/imscc_chunks/chunks.jsonl``.
        3. ``<trainforge_dir>/corpus/chunks.jsonl`` (legacy layout).
        4. ``<trainforge_dir>/chunks.jsonl`` (legacy layout).
        """
        candidates: List[Path] = []
        course_path = self._resolve_libv2_course_path()
        if course_path is not None:
            candidates.append(course_path / "imscc_chunks" / "chunks.jsonl")
        tdir = self._resolve_trainforge_dir()
        if tdir is not None:
            candidates.append(tdir / "imscc_chunks" / "chunks.jsonl")
            candidates.append(tdir / "corpus" / "chunks.jsonl")
            candidates.append(tdir / "chunks.jsonl")
        for cand in candidates:
            if cand.exists():
                rows = self._read_jsonl(cand)
                if rows:
                    return rows
        return []

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Best-effort load of the assessments.json questions[] array.

        Canonical surface: ``<trainforge_dir>/assessments.json`` (the
        single-document JSON written by
        ``MCP/tools/pipeline_tools.py::_generate_assessments``).
        Post-archival, the same file lives at
        ``<libv2_course>/training_specs/assessments.json``.

        Priority chain:

        1. ``<trainforge_dir>/assessments.json``.
        2. ``<trainforge_dir>/training_specs/assessments.json``.
        3. ``<libv2_course>/training_specs/assessments.json``.
        4. ``<libv2_course>/assessments.json``.
        """
        candidates: List[Path] = []
        tdir = self._resolve_trainforge_dir()
        if tdir is not None:
            candidates.append(tdir / "assessments.json")
            candidates.append(tdir / "training_specs" / "assessments.json")
        course_path = self._resolve_libv2_course_path()
        if course_path is not None:
            candidates.append(
                course_path / "training_specs" / "assessments.json"
            )
            candidates.append(course_path / "assessments.json")
        for cand in candidates:
            if cand.exists():
                payload = self._read_json(cand)
                if not isinstance(payload, Mapping):
                    continue
                questions = payload.get("questions")
                if isinstance(questions, list):
                    return [q for q in questions if isinstance(q, Mapping)]
        return []

    def _load_pairs(self) -> Iterable[Dict[str, Any]]:
        """Best-effort load of instruction + preference pairs.

        Reads both ``instruction_pairs.jsonl`` and
        ``preference_pairs.jsonl`` from the trainforge_dir or LibV2
        course's ``training_specs/`` subdir. Yields each row.
        """
        roots: List[Path] = []
        tdir = self._resolve_trainforge_dir()
        if tdir is not None:
            roots.append(tdir / "training_specs")
        course_path = self._resolve_libv2_course_path()
        if course_path is not None:
            roots.append(course_path / "training_specs")
        seen_paths: Set[Path] = set()
        for root in roots:
            for fname in ("instruction_pairs.jsonl", "preference_pairs.jsonl"):
                p = root / fname
                if p in seen_paths or not p.exists():
                    continue
                seen_paths.add(p)
                yield from self._read_jsonl(p)

    # ------------------------------------------------------------------
    # Helpers — reading
    # ------------------------------------------------------------------
    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        """Best-effort JSON read; returns ``None`` on any failure."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "coverage_map: cannot read %s: %s", path, exc,
            )
            return None

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        """Best-effort JSONL read; skips malformed rows, returns the rest."""
        out: List[Dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "coverage_map: cannot read %s: %s", path, exc,
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
    # Helpers — id / refs extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _objective_id(lo: Mapping[str, Any]) -> str:
        """Return the canonical LO id (matches abcd_objective._lo_id)."""
        raw = lo.get("id") or lo.get("objective_id") or ""
        return str(raw).strip()

    @staticmethod
    def _chunk_id(chunk: Mapping[str, Any]) -> str:
        """Resolve the chunk id (Trainforge chunks carry `id` OR `chunk_id`)."""
        raw = chunk.get("id") or chunk.get("chunk_id") or ""
        return str(raw).strip()

    @staticmethod
    def _question_id(question: Mapping[str, Any]) -> str:
        """Resolve the question id (canonical: `question_id`; fallback `id`)."""
        raw = (
            question.get("question_id")
            or question.get("id")
            or question.get("qid")
            or ""
        )
        return str(raw).strip()

    @staticmethod
    def _question_objective_ids(question: Mapping[str, Any]) -> List[str]:
        """Pull objective_id (single) + objective_ids (plural) from a question."""
        out: List[str] = []
        single = question.get("objective_id")
        if isinstance(single, str) and single.strip():
            out.append(single.strip())
        plural = question.get("objective_ids")
        if isinstance(plural, list):
            for ref in plural:
                if isinstance(ref, str) and ref.strip() and ref.strip() not in out:
                    out.append(ref.strip())
        return out

    @staticmethod
    def _question_source_chunks(question: Mapping[str, Any]) -> List[str]:
        """Pull `source_chunks` / `source_chunk_ids` from a question."""
        out: List[str] = []
        for key in ("source_chunks", "source_chunk_ids"):
            val = question.get(key)
            if not isinstance(val, list):
                continue
            for entry in val:
                if isinstance(entry, str) and entry.strip():
                    s = entry.strip()
                    if s not in out:
                        out.append(s)
        return out

    @staticmethod
    def _pair_id(pair: Mapping[str, Any]) -> str:
        """Resolve the pair id."""
        raw = pair.get("id") or pair.get("pair_id") or ""
        return str(raw).strip()

    @staticmethod
    def _normalize_refs(raw_refs: Any) -> List[str]:
        """Coerce a refs[] field to a deduped, stripped list of strings."""
        if not isinstance(raw_refs, (list, tuple)):
            return []
        out: List[str] = []
        for r in raw_refs:
            if isinstance(r, str) and r.strip() and r.strip() not in out:
                out.append(r.strip())
        return out

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(
        objectives_list: Sequence[Mapping[str, Any]],
        *,
        orphan_chunks: Sequence[str],
        orphan_questions: Sequence[str],
    ) -> Dict[str, Any]:
        """Compute the summary block."""
        total = len(objectives_list)
        with_chunks = sum(
            1 for o in objectives_list if o.get("chunks")
        )
        with_questions = sum(
            1 for o in objectives_list if o.get("questions")
        )
        with_training_pairs = sum(
            1 for o in objectives_list if o.get("training_pairs")
        )
        orphan_objectives = [
            o["objective_id"]
            for o in objectives_list
            if not o.get("chunks")
            and not o.get("questions")
            and not o.get("training_pairs")
        ]
        # Dedupe orphan lists; sort for byte-stability across runs.
        return {
            "total_objectives": total,
            "objectives_with_chunks": with_chunks,
            "objectives_with_questions": with_questions,
            "objectives_with_training_pairs": with_training_pairs,
            "orphan_objectives": sorted(set(orphan_objectives)),
            "orphan_chunks": sorted(set(orphan_chunks)),
            "orphan_questions": sorted(set(orphan_questions)),
        }

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------
    def _emit_aggregator_decision(self, report: Mapping[str, Any]) -> None:
        """Emit one ``coverage_map_aggregated`` decision per build.

        Best-effort: any failure in the capture path is logged and
        swallowed. Rationale interpolates the six summary counts so it
        clears the 20-char floor and stays replayable post-hoc.
        """
        capture = self.decision_capture
        if capture is None:
            return
        try:
            summary = report.get("summary") or {}
            total_objectives = summary.get("total_objectives", 0)
            with_chunks = summary.get("objectives_with_chunks", 0)
            with_questions = summary.get("objectives_with_questions", 0)
            with_pairs = summary.get("objectives_with_training_pairs", 0)
            orphan_objectives = summary.get("orphan_objectives") or []
            orphan_chunks = summary.get("orphan_chunks") or []
            orphan_questions = summary.get("orphan_questions") or []

            rationale = (
                f"Coverage-map aggregated: total_objectives={total_objectives}, "
                f"objectives_with_chunks={with_chunks}, "
                f"objectives_with_questions={with_questions}, "
                f"objectives_with_training_pairs={with_pairs}, "
                f"orphan_objectives={len(orphan_objectives)}, "
                f"orphan_chunks={len(orphan_chunks)}, "
                f"orphan_questions={len(orphan_questions)}"
            )
            capture.log_decision(
                decision_type="coverage_map_aggregated",
                decision=(
                    f"orphan_objectives={len(orphan_objectives)}/"
                    f"{total_objectives}"
                ),
                rationale=rationale,
                context=str({
                    "course_code": report.get("course_code"),
                    "run_id": report.get("run_id"),
                    "schema_version": report.get("schema_version"),
                }),
                metrics={
                    "total_objectives": total_objectives,
                    "objectives_with_chunks": with_chunks,
                    "objectives_with_questions": with_questions,
                    "objectives_with_training_pairs": with_pairs,
                    "orphan_objective_count": len(orphan_objectives),
                    "orphan_chunk_count": len(orphan_chunks),
                    "orphan_question_count": len(orphan_questions),
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "coverage_map_aggregated: %s",
                exc,
            )


__all__ = [
    "CoverageMapAggregator",
    "SCHEMA_VERSION",
]
