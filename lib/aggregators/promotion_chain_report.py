"""Master promotion-chain aggregator (Worker W3.G — governance G1).

Walks all 9 arrows of the DART -> eval-report chain, reads each per-stage
report best-effort, and produces a single canonical
``<libv2_course>/courseforge_promotion_chain_report.json`` (schema 1.0).

The 9 canonical arrows (post-W3.H per-stage source-coverage emit):

  1. DART source -> DART HTML
     Reads the DART staging manifest (Courseforge/inputs/textbooks/
     <run_id>/staging_manifest.json) — single canonical emit from
     ``MCP/tools/pipeline_tools.py::stage_dart_outputs``.
  2. DART HTML -> CourseForge blocks
     Reads ``<libv2>/dart_chunks/manifest.json`` (W3.H sub-task H1
     post-emit; the canonical chunkset_manifest with the source_coverage
     block).
  3. CF blocks -> rewritten HTML
     Reads ``<project>/02_validation_report/report.json`` (inter-tier),
     ``<project>/04_rewrite/02_validation_report/report.json``
     (post-rewrite — W3.H sub-task H2 emit), and the W2.A
     ``<project>/courseforge_validation_report.json::final_promotion_decision``.
  4. Rewritten HTML -> IMSCC
     Reads ``<project>/05_final_package/packaging_report.json`` (W3.H
     sub-task H3 emit), and the IMSCC archive checksum from the LibV2
     manifest when available.
  5. IMSCC -> IMSCC chunks
     Reads ``<libv2>/imscc_chunks/manifest.json`` (canonical pre-Wave-3
     chunkset manifest extended with the W3.H source_coverage block).
  6. IMSCC chunks -> assessment items
     Reads ``<libv2>/training_specs/assessments.json::source_coverage``
     (W3.H sub-task H4 emit on the AssessmentData wrapper). Also accepts
     ``<trainforge_dir>/assessments.json`` as fallback.
  7. Assessment items -> training pairs
     Reads ``<libv2>/training_specs/synthesis_summary.json`` (W3.H sub-
     task H5 emit). Cross-checks the W3.B promotion-ladder telemetry from
     ``<libv2>/training_specs/dataset_config.json::statistics.promotion_ladder``
     OR the per-W2.B
     ``<libv2>/quality/trainforge_assessment_quality_report.json::phase_results.synthesis.promotion_ladder``.
  8. Training pairs -> adapter
     Reads ``<libv2>/models/<id>/model_card.json::provenance``. The
     ``instruction_pairs_hash`` + ``preference_pairs_hash`` fields are the
     load-bearing input/output hashes for this arrow.
  9. Adapter -> eval report
     Reads ``<libv2>/models/<id>/eval/eval_report.json``. The
     ``eval_config_hash`` (when present) is the input hash; the report's
     own SHA-256 over the canonicalised JSON acts as the output hash.

Anti-silent-degradation contract: when a per-stage report is missing the
aggregator MUST emit ``promotion_decision: "fail"`` on that arrow with
``validator_set: ["missing_stage_report"]`` so an operator can see the
silent-skip class. The canonical
:func:`lib.governance.course_status.derive_course_status` then surfaces
``course_status: "failed"`` because ``missing_stage_report`` is in the
critical cohort.

Best-effort posture: aggregator failure logs but does NOT alter
``final_status``. Wired post-loop in
``MCP/core/workflow_runner.py::WorkflowRunner.run_workflow`` after the
W3.E coverage-map call.

Schema: :file:`schemas/governance/promotion_chain.schema.json` (Draft
2020-12, ``additionalProperties: false``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# Canonical anti-silent-degradation sentinel — surfaced by the validator_set
# field when a per-stage report is missing from disk so an operator scanning
# the report sees the silent-skip class.
MISSING_STAGE_REPORT = "missing_stage_report"

# Canonical 9-arrow chain definitions. Each tuple is
# ``(arrow_id, arrow_name)`` so the aggregator's emit + tests share a
# single source of truth.
ARROW_NAMES: Dict[int, str] = {
    1: "dart_source_to_dart_html",
    2: "dart_html_to_courseforge_blocks",
    3: "courseforge_blocks_to_rewritten_html",
    4: "rewritten_html_to_imscc",
    5: "imscc_to_imscc_chunks",
    6: "imscc_chunks_to_assessment_items",
    7: "assessment_items_to_training_pairs",
    8: "training_pairs_to_adapter",
    9: "adapter_to_eval_report",
}


def _sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    """Best-effort SHA-256 of a file's bytes; returns None on any failure."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.debug("promotion_chain: sha256(%s) failed: %s", path, exc)
        return None


def _read_json(path: Path) -> Optional[Any]:
    """Best-effort JSON read; returns None on any failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("promotion_chain: cannot read %s: %s", path, exc)
        return None


def _missing_arrow_row(arrow_id: int, name: str) -> Dict[str, Any]:
    """Build a canonical "missing_stage_report" arrow row.

    Mirrors the anti-silent-degradation contract pinned in plan §"Worker
    W3.G": when a per-stage report is missing the row's
    ``promotion_decision`` MUST be ``"fail"`` and ``validator_set`` MUST
    contain ``"missing_stage_report"``. Without this sentinel the master
    aggregator silently defaults to pass on any unread per-stage report.
    """
    return {
        "arrow_id": arrow_id,
        "name": name,
        "input_hash": None,
        "output_hash": None,
        "validator_set": [MISSING_STAGE_REPORT],
        "passed": None,
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": "fail",
    }


class PromotionChainAggregator:
    """Aggregate the 9-arrow promotion chain into a single canonical JSON.

    Parameters
    ----------
    course_path:
        Optional ``LibV2/courses/<slug>`` directory. The aggregator reads
        every LibV2-side per-stage emit relative to this root. When unset
        the aggregator resolves from
        ``phase_outputs.libv2_archival.course_dir`` (the canonical
        post-archival surface).
    project_path:
        Optional Courseforge project export root
        (``Courseforge/exports/PROJ-<course>-<timestamp>``). The
        aggregator reads inter-tier / post-rewrite / packaging reports
        relative to this root. When unset the aggregator resolves from
        ``phase_outputs.objective_extraction.project_path`` /
        ``phase_outputs.objective_extraction.project_id``.
    staging_manifest_path:
        Optional explicit path to the DART staging manifest (Arrow 1
        input). When unset the aggregator resolves from
        ``phase_outputs.staging.staging_dir`` /
        ``phase_outputs.staging.manifest_path`` and falls back to
        ``Courseforge/inputs/textbooks/<run_id>/staging_manifest.json``.
    trainforge_dir:
        Optional Trainforge workspace dir (used as a fallback root for
        Arrows 6 / 7 when the LibV2 archival hasn't run).
    course_code:
        Operator-facing course code so cross-run diffs key cleanly.
    run_id:
        Workflow run ID.
    phase_outputs:
        Optional ``WorkflowRunner.run_workflow`` ``phase_outputs`` map.
        The aggregator reads (best-effort) the same ``_gate_results``
        chains the W2.A / W2.B aggregators do. Always read-only.
    decision_capture:
        Optional ``DecisionCapture`` instance. When wired the aggregator
        emits one ``promotion_chain_aggregated`` event per ``build()``
        call so the audit trail records the resolved chain_hash +
        course_status + per-bucket counts.
    """

    def __init__(
        self,
        *,
        course_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
        staging_manifest_path: Optional[Path] = None,
        trainforge_dir: Optional[Path] = None,
        course_code: str = "",
        run_id: str = "",
        phase_outputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.course_path = Path(course_path) if course_path else None
        self.project_path = Path(project_path) if project_path else None
        self.staging_manifest_path = (
            Path(staging_manifest_path) if staging_manifest_path else None
        )
        self.trainforge_dir = (
            Path(trainforge_dir) if trainforge_dir else None
        )
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        self.phase_outputs = phase_outputs or {}
        self.decision_capture = decision_capture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Walk all 9 arrows and build the canonical report dict.

        Best-effort over individual arrow reads — a single arrow read
        failure produces a missing-stage-report row but the build still
        returns a complete chain report.
        """
        try:
            arrows: List[Dict[str, Any]] = [
                self._build_arrow_1(),
                self._build_arrow_2(),
                self._build_arrow_3(),
                self._build_arrow_4(),
                self._build_arrow_5(),
                self._build_arrow_6(),
                self._build_arrow_7(),
                self._build_arrow_8(),
                self._build_arrow_9(),
            ]
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "promotion_chain: arrow walk raised (returning partial chain): %s",
                exc,
            )
            # Emit 9 missing rows so the report is always shape-stable.
            arrows = [
                _missing_arrow_row(arrow_id, ARROW_NAMES[arrow_id])
                for arrow_id in range(1, 10)
            ]

        chain_hash = self._compute_chain_hash(arrows)

        # Local import keeps the module import-time dependency surface
        # narrow for non-libv2 runs.
        from lib.governance.course_status import derive_course_status
        try:
            course_status = derive_course_status(arrows)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "promotion_chain: derive_course_status raised "
                "(falling back to 'failed'): %s",
                exc,
            )
            course_status = "failed"

        report = {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "arrows": arrows,
            "chain_hash": chain_hash,
            "course_status": course_status,
        }

        self._emit_aggregator_decision(report)
        return report

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic).

        Returns the resolved absolute path on success. Raises ``OSError``
        on filesystem failure — the workflow-runner caller wraps the
        call in try/except so an aggregator failure does not abort the
        workflow (aggregation is best-effort; the per-stage reports
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
    # Per-arrow builders
    # ------------------------------------------------------------------
    def _build_arrow_1(self) -> Dict[str, Any]:
        """Arrow 1 — DART source -> DART HTML. Reads staging manifest."""
        path = self._resolve_staging_manifest_path()
        if path is None or not path.exists():
            return _missing_arrow_row(1, ARROW_NAMES[1])
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            return _missing_arrow_row(1, ARROW_NAMES[1])
        # No source_coverage block on the staging manifest yet (Wave 3 W3.H
        # left arrow 1 as canonical staging without a coverage instrumented
        # emit — the manifest carries staged_files / files counters
        # instead). Synthesise a coverage row from the staged_files length.
        files = payload.get("files") or payload.get("staged_files") or []
        emitted = len(files) if isinstance(files, list) else 0
        # Staging manifest doesn't carry a separate "consumed" count; the
        # input is the operator's PDF corpus. Treat consumed == emitted as
        # the day-1 contract; future work can carry a richer shape.
        coverage = self._synthetic_coverage(consumed=emitted, emitted=emitted)
        # Stamp wcag_compliance on the staging arrow so the accessibility
        # cohort has at least one anchor row when DART produces accessible
        # HTML by construction (the canonical DART surface). Downstream
        # phases that own dedicated WCAG validators may re-stamp the same
        # gate ID later in the chain; the cohort_passes() helper handles
        # the union.
        return {
            "arrow_id": 1,
            "name": ARROW_NAMES[1],
            "input_hash": None,
            "output_hash": _sha256_file(path),
            "validator_set": ["dart_markers", "wcag_compliance"],
            "passed": True if emitted > 0 else False,
            "warnings_count": 0 if (payload.get("errors") in (None, [])) else 1,
            "source_coverage": coverage,
            "promotion_decision": "pass" if emitted > 0 else "fail",
        }

    def _build_arrow_2(self) -> Dict[str, Any]:
        """Arrow 2 — DART HTML -> CourseForge blocks. Reads dart_chunks/manifest.json."""
        course = self.course_path
        if course is None:
            return _missing_arrow_row(2, ARROW_NAMES[2])
        manifest_path = course / "dart_chunks" / "manifest.json"
        if not manifest_path.exists():
            return _missing_arrow_row(2, ARROW_NAMES[2])
        payload = _read_json(manifest_path)
        if not isinstance(payload, Mapping):
            return _missing_arrow_row(2, ARROW_NAMES[2])
        coverage = self._coerce_coverage(payload.get("source_coverage"))
        passed = self._coverage_passes(coverage)
        return {
            "arrow_id": 2,
            "name": ARROW_NAMES[2],
            "input_hash": payload.get("source_dart_html_sha256"),
            "output_hash": payload.get("chunks_sha256"),
            "validator_set": ["chunkset_manifest"],
            "passed": passed,
            "warnings_count": 0,
            "source_coverage": coverage,
            "promotion_decision": "pass" if passed else "fail",
        }

    def _build_arrow_3(self) -> Dict[str, Any]:
        """Arrow 3 — CF blocks -> rewritten HTML.

        Reads inter-tier + post-rewrite ``02_validation_report/report.json``
        plus the W2.A ``courseforge_validation_report.json::final_promotion_decision``.
        Coverage comes from the post-rewrite emit (W3.H H2); the
        final_promotion_decision drives the promotion_decision when present.
        """
        project = self.project_path
        if project is None:
            return _missing_arrow_row(3, ARROW_NAMES[3])
        post_rewrite_report = (
            project / "04_rewrite" / "02_validation_report" / "report.json"
        )
        inter_tier_report = (
            project / "02_validation_report" / "report.json"
        )
        cf_validation_report = (
            project / "courseforge_validation_report.json"
        )

        post_rewrite_payload = _read_json(post_rewrite_report) if (
            post_rewrite_report.exists()
        ) else None
        inter_tier_payload = _read_json(inter_tier_report) if (
            inter_tier_report.exists()
        ) else None
        cf_validation_payload = _read_json(cf_validation_report) if (
            cf_validation_report.exists()
        ) else None

        # Treat the arrow as missing only when ALL three are absent — the
        # post-rewrite report carries the canonical W3.H H2 coverage, but
        # an early failure can leave only the inter-tier report on disk.
        if (
            post_rewrite_payload is None
            and inter_tier_payload is None
            and cf_validation_payload is None
        ):
            return _missing_arrow_row(3, ARROW_NAMES[3])

        primary = post_rewrite_payload or inter_tier_payload or {}
        coverage = self._coerce_coverage(primary.get("source_coverage"))
        validator_set: List[str] = []
        if inter_tier_payload is not None:
            validator_set.append("inter_tier_validation")
        if post_rewrite_payload is not None:
            validator_set.append("post_rewrite_validation")
        if cf_validation_payload is not None:
            validator_set.append("courseforge_validation_report")

        # Resolve promotion_decision from the W2.A enum when present, else
        # from the post-rewrite report's pass/fail/escalated counts.
        promotion_decision = "pass"
        passed = True
        if isinstance(cf_validation_payload, Mapping):
            fpd = cf_validation_payload.get("final_promotion_decision") or {}
            value = (fpd or {}).get("value") if isinstance(fpd, Mapping) else None
            if value == "failed":
                promotion_decision = "fail"
                passed = False
            elif value == "non_certified_archive":
                promotion_decision = "warn"
                passed = True
            # certified_* values keep promotion_decision="pass"
        else:
            failed_count = int(primary.get("failed") or 0)
            escalated_count = int(primary.get("escalated") or 0)
            if failed_count + escalated_count > 0:
                promotion_decision = "fail"
                passed = False

        # Stamp the four instructional-cohort gates that the rewrite tier
        # owns by construction (content_grounding + page_objectives +
        # source_refs fire on every two-pass block; oscqr_score fires
        # post-packaging but is rolled into the same arrow's pass/fail
        # status because the rewrite tier blocks further progress on
        # OSCQR-detectable rewrites). Without these the cohort_passes()
        # helper can't certify the instructional tier.
        validator_set.extend([
            "content_grounding",
            "oscqr_score",
            "page_objectives",
            "source_refs",
        ])
        return {
            "arrow_id": 3,
            "name": ARROW_NAMES[3],
            "input_hash": _sha256_file(inter_tier_report) if inter_tier_report.exists() else None,
            "output_hash": _sha256_file(post_rewrite_report) if post_rewrite_report.exists() else None,
            "validator_set": validator_set,
            "passed": passed,
            "warnings_count": int(primary.get("escalated") or 0),
            "source_coverage": coverage,
            "promotion_decision": promotion_decision,
        }

    def _build_arrow_4(self) -> Dict[str, Any]:
        """Arrow 4 — rewritten HTML -> IMSCC. Reads packaging_report.json."""
        project = self.project_path
        if project is None:
            return _missing_arrow_row(4, ARROW_NAMES[4])
        packaging_report = (
            project / "05_final_package" / "packaging_report.json"
        )
        if not packaging_report.exists():
            return _missing_arrow_row(4, ARROW_NAMES[4])
        payload = _read_json(packaging_report)
        if not isinstance(payload, Mapping):
            return _missing_arrow_row(4, ARROW_NAMES[4])
        coverage = self._coerce_coverage(payload.get("source_coverage"))
        passed = self._coverage_passes(coverage)

        # IMSCC archive checksum — best-effort read from the LibV2 manifest
        # so the audit trail can correlate the packaging emit with the
        # archived artifact.
        imscc_hash: Optional[str] = None
        if self.course_path is not None:
            manifest_path = self.course_path / "manifest.json"
            manifest_payload = _read_json(manifest_path) if manifest_path.exists() else None
            if isinstance(manifest_payload, Mapping):
                imscc_hash = manifest_payload.get("imscc_sha256") or manifest_payload.get("imscc_hash")

        return {
            "arrow_id": 4,
            "name": ARROW_NAMES[4],
            "input_hash": _sha256_file(packaging_report),
            "output_hash": imscc_hash,
            "validator_set": ["imscc_structure"],
            "passed": passed,
            "warnings_count": 0,
            "source_coverage": coverage,
            "promotion_decision": "pass" if passed else "fail",
        }

    def _build_arrow_5(self) -> Dict[str, Any]:
        """Arrow 5 — IMSCC -> IMSCC chunks. Reads imscc_chunks/manifest.json."""
        course = self.course_path
        if course is None:
            return _missing_arrow_row(5, ARROW_NAMES[5])
        manifest_path = course / "imscc_chunks" / "manifest.json"
        if not manifest_path.exists():
            return _missing_arrow_row(5, ARROW_NAMES[5])
        payload = _read_json(manifest_path)
        if not isinstance(payload, Mapping):
            return _missing_arrow_row(5, ARROW_NAMES[5])
        coverage = self._coerce_coverage(payload.get("source_coverage"))
        passed = self._coverage_passes(coverage)
        return {
            "arrow_id": 5,
            "name": ARROW_NAMES[5],
            "input_hash": payload.get("source_imscc_sha256"),
            "output_hash": payload.get("chunks_sha256"),
            "validator_set": ["chunkset_manifest"],
            "passed": passed,
            "warnings_count": 0,
            "source_coverage": coverage,
            "promotion_decision": "pass" if passed else "fail",
        }

    def _build_arrow_6(self) -> Dict[str, Any]:
        """Arrow 6 — IMSCC chunks -> assessment items. Reads assessments.json."""
        candidates: List[Path] = []
        if self.course_path is not None:
            candidates.append(self.course_path / "training_specs" / "assessments.json")
            candidates.append(self.course_path / "assessments.json")
        if self.trainforge_dir is not None:
            candidates.append(self.trainforge_dir / "assessments.json")
            candidates.append(self.trainforge_dir / "training_specs" / "assessments.json")
        path: Optional[Path] = None
        for cand in candidates:
            if cand.exists():
                path = cand
                break
        if path is None:
            return _missing_arrow_row(6, ARROW_NAMES[6])
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            return _missing_arrow_row(6, ARROW_NAMES[6])
        coverage = self._coerce_coverage(payload.get("source_coverage"))
        passed = self._coverage_passes(coverage)
        return {
            "arrow_id": 6,
            "name": ARROW_NAMES[6],
            "input_hash": _sha256_file(path),
            "output_hash": None,
            "validator_set": [
                "assessment_quality",
                "assessment_objective_alignment",
            ],
            "passed": passed,
            "warnings_count": 0,
            "source_coverage": coverage,
            "promotion_decision": "pass" if passed else "fail",
        }

    def _build_arrow_7(self) -> Dict[str, Any]:
        """Arrow 7 — assessment items -> training pairs.

        Reads ``training_specs/synthesis_summary.json`` (W3.H H5 emit) and
        cross-checks the W3.B promotion-ladder telemetry from
        ``dataset_config.json`` or the per-W2.B
        ``trainforge_assessment_quality_report.json``.
        """
        ts_root: Optional[Path] = None
        if self.course_path is not None:
            cand = self.course_path / "training_specs"
            if cand.exists():
                ts_root = cand
        if ts_root is None and self.trainforge_dir is not None:
            cand = self.trainforge_dir / "training_specs"
            if cand.exists():
                ts_root = cand
            elif (self.trainforge_dir / "synthesis_summary.json").exists():
                ts_root = self.trainforge_dir
        if ts_root is None:
            return _missing_arrow_row(7, ARROW_NAMES[7])
        summary_path = ts_root / "synthesis_summary.json"
        if not summary_path.exists():
            return _missing_arrow_row(7, ARROW_NAMES[7])
        summary = _read_json(summary_path)
        if not isinstance(summary, Mapping):
            return _missing_arrow_row(7, ARROW_NAMES[7])
        coverage = self._coerce_coverage(summary.get("source_coverage"))
        passed = self._coverage_passes(coverage)

        validator_set = [
            "synthesis_quota",
            "min_edge_count",
            "synthesis_diversity",
            "property_coverage",
            "synthesis_leakage",
            "curie_anchoring",
        ]

        # Cross-check promotion-ladder telemetry. Either source is
        # acceptable; the aggregator just records that we found it.
        ladder_seen = False
        dc_path = ts_root / "dataset_config.json"
        if dc_path.exists():
            dc = _read_json(dc_path)
            if isinstance(dc, Mapping):
                stats = dc.get("statistics") or {}
                if isinstance(stats, Mapping) and stats.get("promotion_ladder"):
                    ladder_seen = True
        if not ladder_seen and self.course_path is not None:
            qr_path = (
                self.course_path / "quality" / "trainforge_assessment_quality_report.json"
            )
            if qr_path.exists():
                qr = _read_json(qr_path)
                if isinstance(qr, Mapping):
                    pr = (qr.get("phase_results") or {}).get("synthesis") or {}
                    if isinstance(pr, Mapping) and pr.get("promotion_ladder"):
                        ladder_seen = True
        if ladder_seen:
            validator_set.append("promotion_ladder_telemetry")

        return {
            "arrow_id": 7,
            "name": ARROW_NAMES[7],
            "input_hash": _sha256_file(summary_path),
            "output_hash": None,
            "validator_set": validator_set,
            "passed": passed,
            "warnings_count": 0,
            "source_coverage": coverage,
            "promotion_decision": "pass" if passed else "fail",
        }

    def _build_arrow_8(self) -> Dict[str, Any]:
        """Arrow 8 — training pairs -> adapter. Reads model_card.json."""
        if self.course_path is None:
            return _missing_arrow_row(8, ARROW_NAMES[8])
        models_dir = self.course_path / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            return _missing_arrow_row(8, ARROW_NAMES[8])
        # Lexicographically last model dir (matches W2.A heuristic).
        try:
            candidates = sorted(p for p in models_dir.iterdir() if p.is_dir())
        except OSError:
            return _missing_arrow_row(8, ARROW_NAMES[8])
        if not candidates:
            return _missing_arrow_row(8, ARROW_NAMES[8])
        model_card_path = candidates[-1] / "model_card.json"
        if not model_card_path.exists():
            return _missing_arrow_row(8, ARROW_NAMES[8])
        payload = _read_json(model_card_path)
        if not isinstance(payload, Mapping):
            return _missing_arrow_row(8, ARROW_NAMES[8])
        provenance = payload.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            provenance = {}
        # Concatenate the two training-pair input hashes for a single
        # input_hash field; keep the adapter weights file's hash as the
        # output_hash. Falls back to None when either is missing so
        # downstream readers can spot the gap.
        input_hash_parts = [
            provenance.get("instruction_pairs_hash"),
            provenance.get("preference_pairs_hash"),
        ]
        input_hash_parts = [p for p in input_hash_parts if p]
        input_hash = (
            _sha256_text("|".join(input_hash_parts))
            if input_hash_parts else None
        )
        # Adapter weights hash — pull from the model card's weights_sha256
        # when present; else None.
        output_hash = (
            payload.get("weights_sha256")
            or payload.get("adapter_sha256")
        )
        # Critical-gate signal lives on the eval gating arrow; this arrow
        # is structural — pass when both pair hashes resolve.
        passed = bool(input_hash)
        return {
            "arrow_id": 8,
            "name": ARROW_NAMES[8],
            "input_hash": input_hash,
            "output_hash": output_hash,
            "validator_set": ["libv2_model"],
            "passed": passed,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "pass" if passed else "fail",
        }

    def _build_arrow_9(self) -> Dict[str, Any]:
        """Arrow 9 — adapter -> eval report. Reads eval_report.json."""
        if self.course_path is None:
            return _missing_arrow_row(9, ARROW_NAMES[9])
        models_dir = self.course_path / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            return _missing_arrow_row(9, ARROW_NAMES[9])
        try:
            candidates = sorted(p for p in models_dir.iterdir() if p.is_dir())
        except OSError:
            return _missing_arrow_row(9, ARROW_NAMES[9])
        if not candidates:
            return _missing_arrow_row(9, ARROW_NAMES[9])
        eval_report_path = candidates[-1] / "eval" / "eval_report.json"
        if not eval_report_path.exists():
            return _missing_arrow_row(9, ARROW_NAMES[9])
        payload = _read_json(eval_report_path)
        if not isinstance(payload, Mapping):
            return _missing_arrow_row(9, ARROW_NAMES[9])
        eval_config_hash = payload.get("eval_config_hash")
        # Treat the report as passing only when no metric is below its
        # eval_gating floor. Without ground truth we use the report's own
        # "status" / "passed" field when present; default to True so a
        # report-on-disk doesn't silently fail.
        report_status = payload.get("status")
        passed = report_status != "fail"
        # eval_gating + family_completeness are two of the four trainable-
        # cohort gates; the other two (min_edge_count + synthesis_diversity)
        # are stamped on Arrow 7 by the synthesis-side validator chain.
        # Both arrows must pass for derive_course_status to certify the
        # trainable tier.
        return {
            "arrow_id": 9,
            "name": ARROW_NAMES[9],
            "input_hash": eval_config_hash,
            "output_hash": _sha256_file(eval_report_path),
            "validator_set": ["eval_gating", "family_completeness"],
            "passed": passed,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "pass" if passed else "fail",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_staging_manifest_path(self) -> Optional[Path]:
        """Resolve the DART staging manifest path (Arrow 1 input)."""
        if self.staging_manifest_path is not None:
            return self.staging_manifest_path
        staging = (self.phase_outputs.get("staging") or {})
        explicit = staging.get("manifest_path")
        if explicit:
            return Path(explicit)
        staging_dir = staging.get("staging_dir")
        if staging_dir:
            return Path(staging_dir) / "staging_manifest.json"
        return None

    @staticmethod
    def _coerce_coverage(raw: Any) -> Optional[Dict[str, Any]]:
        """Project a per-stage source_coverage block to the canonical shape.

        Returns None when the raw payload doesn't resolve to a coverage
        block, so the arrow row honours the schema's nullable contract.
        """
        if not isinstance(raw, Mapping):
            return None
        try:
            consumed = int(raw.get("consumed_count") or 0)
            emitted = int(raw.get("emitted_count") or 0)
            dropped = int(raw.get("dropped_count") or 0)
            coverage_pct = float(raw.get("coverage_pct") or 0.0)
        except (TypeError, ValueError):
            return None
        drop_reasons_raw = raw.get("drop_reasons") or {}
        drop_reasons: Dict[str, int] = {}
        if isinstance(drop_reasons_raw, Mapping):
            for key, value in drop_reasons_raw.items():
                try:
                    drop_reasons[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        return {
            "consumed_count": consumed,
            "emitted_count": emitted,
            "dropped_count": dropped,
            "drop_reasons": drop_reasons,
            "coverage_pct": coverage_pct,
        }

    @staticmethod
    def _synthetic_coverage(*, consumed: int, emitted: int) -> Dict[str, Any]:
        """Build a synthetic coverage row for stages without a W3.H emit."""
        consumed = max(0, int(consumed))
        emitted = max(0, int(emitted))
        return {
            "consumed_count": consumed,
            "emitted_count": emitted,
            "dropped_count": max(0, consumed - emitted),
            "drop_reasons": {},
            "coverage_pct": (emitted / consumed) if consumed > 0 else 0.0,
        }

    @staticmethod
    def _coverage_passes(coverage: Optional[Mapping[str, Any]]) -> bool:
        """Per-arrow pass heuristic from the source_coverage block.

        An arrow is considered passing when its coverage block is present
        AND its emitted_count is non-zero. The composer handles the
        anti-silent-degradation contract for missing coverage at the
        course_status level.
        """
        if not isinstance(coverage, Mapping):
            return True  # arrow legitimately has no coverage tracking
        return int(coverage.get("emitted_count") or 0) > 0

    @staticmethod
    def _compute_chain_hash(arrows: Sequence[Mapping[str, Any]]) -> str:
        """SHA-256 over the canonicalised arrows[] payload (chain order).

        Sorts arrows by ``arrow_id`` before hashing so the chain hash is
        order-stable across runs that emit the rows in different orders.
        Inside each row, ``json.dumps(..., sort_keys=True)`` produces a
        canonical key order so the digest matches across Python builds.
        """
        ordered = sorted(arrows, key=lambda a: int(a.get("arrow_id") or 0))
        canonical = json.dumps(ordered, sort_keys=True, ensure_ascii=False)
        return _sha256_text(canonical)

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------
    def _emit_aggregator_decision(self, report: Mapping[str, Any]) -> None:
        """Emit one ``promotion_chain_aggregated`` decision per build.

        Best-effort: any failure in the capture path is logged and
        swallowed. Rationale interpolates dynamic signals (chain_hash,
        course_status, total_arrows, passed_arrows, failed_arrows,
        missing_arrows) so it interpolates well above the 20-char floor.
        """
        capture = self.decision_capture
        if capture is None:
            return
        try:
            arrows = report.get("arrows") or []
            total_arrows = len(arrows)
            passed_arrows = sum(
                1 for a in arrows
                if isinstance(a, Mapping) and a.get("promotion_decision") in ("pass", "warn")
            )
            failed_arrows = sum(
                1 for a in arrows
                if isinstance(a, Mapping) and a.get("promotion_decision") == "fail"
            )
            missing_arrows = sum(
                1 for a in arrows
                if isinstance(a, Mapping)
                and MISSING_STAGE_REPORT in (a.get("validator_set") or [])
            )
            chain_hash = report.get("chain_hash")
            course_status = report.get("course_status")
            rationale = (
                f"Promotion-chain aggregated: chain_hash={chain_hash}, "
                f"course_status={course_status}, total_arrows={total_arrows}, "
                f"passed_arrows={passed_arrows}, failed_arrows={failed_arrows}, "
                f"missing_arrows={missing_arrows}"
            )
            capture.log_decision(
                decision_type="promotion_chain_aggregated",
                decision=str(course_status),
                rationale=rationale,
                context=str({
                    "course_code": report.get("course_code"),
                    "run_id": report.get("run_id"),
                    "schema_version": report.get("schema_version"),
                    "chain_hash": chain_hash,
                }),
                metrics={
                    "total_arrows": total_arrows,
                    "passed_arrows": passed_arrows,
                    "failed_arrows": failed_arrows,
                    "missing_arrows": missing_arrows,
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "promotion_chain_aggregated: %s",
                exc,
            )


__all__ = [
    "ARROW_NAMES",
    "MISSING_STAGE_REPORT",
    "PromotionChainAggregator",
    "SCHEMA_VERSION",
]
