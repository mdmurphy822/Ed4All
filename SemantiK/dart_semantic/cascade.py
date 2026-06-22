"""Stage 1..13 cascade driver — the v2 pipeline body.

This module owns the end-to-end Stage 1..13 flow described in
``architecture.md`` §2. It is the canonical home of :func:`run_full_cascade`
(previously embedded in ``scripts/run_stage12_smoke.py``); the smoke
harness and the corpus-level eval driver both import it from here, and
:mod:`dart_semantic.pipeline_v2` calls :func:`run_pipeline_v2` to expose
the cascade as the v2 pipeline entry point.

Stage map (matches ``run_full_cascade`` body):

    Stage 1+2  extract + featurize          (deterministic)
    Stage 3    council                      (run_council)
    Stage 4    cross-BERT reranker          (arbitrate)
    Stage 5    structure graph              (build_structure_graph)
    Stage 5b   GLM-OCR table enrichment     (optional, opt-in)
    Stage 6    Qwen region specialists      (run_qwen_specialists)
    Stage 7    per-region HARD gate         (gate_per_region)
    Stage 8    per-region SOFT reranker     (rerank_per_region)
    Stage 9    deterministic assembler      (assemble_document)
    Stage 10   document-level HARD gate     (gate_document)
    Stage 11   document-level SOFT reranker (score_document)
    Stage 12   ThetaEvaluator               (evaluate)
    Stage 13   exit decider + offline retry (maybe_offline_retry/decide_exit)

Single process, single GPU. No PDF-level parallelism — the shared
validator (one Chromium context) and the 8 GB GPU are both single
resources; see ``feedback_qwen_build_serial``.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .assembler import AssemblerConfig, assemble_document
from .council.cross_reranker import arbitrate
from .council.orchestrator import run_council
from .gates import gate_document, gate_per_region, rerank_per_region
from .qwen_specialists.runner import run_qwen_specialists
from .soft_reranker import score_document
from .structure_graph import Region, build_structure_graph
from .theta import decide_exit, evaluate, maybe_offline_retry
from .v2_config import DEFAULT_V2_CONFIG, V2Config
from .validate import HtmlValidator


# ---------------------------------------------------------------------------
# Stage-output aggregation helpers
# ---------------------------------------------------------------------------


def _cap_regions_per_kind(regions: list[Region], *, cap: int) -> list[Region]:
    if cap <= 0:
        return list(regions)
    counts: Counter[str] = Counter()
    out: list[Region] = []
    for r in regions:
        if counts[r.kind] >= cap:
            continue
        out.append(r)
        counts[r.kind] += 1
    return out


def _stage7_violation_distribution(
    results: dict[int, list[Any]],
) -> dict[str, int]:
    """Aggregate axe violation IDs across every per-candidate GateResult.

    Counts each rule once per failing candidate (not per node). A rule
    that fails the same candidate under multiple checks still gets a
    single tally.
    """
    counts: Counter[str] = Counter()
    for cand_results in results.values():
        for gr in cand_results:
            for check in gr.checks:
                if check.passed:
                    continue
                # axe check details have a "violations" list with id/impact.
                vlist = check.details.get("violations") if hasattr(check, "details") else None
                if vlist:
                    for v in vlist:
                        rid = v.get("id") if isinstance(v, dict) else None
                        if rid:
                            counts[rid] += 1
                else:
                    # Non-axe failures recorded under the check enum value.
                    counts[f"non_axe:{check.check.value}"] += 1
    return dict(counts)


def _stage7_skip_distribution(
    results: dict[int, list[Any]],
) -> dict[str, int]:
    """Aggregate per-check skip counts across every per-candidate GateResult.

    A "skip" is a CheckOutcome with ``passed=True`` AND ``skipped=True``
    — the check did not actually measure the candidate (gap-fill
    synthetic regions whose source text is empty, lxml-missing math
    regions, etc.). Returned dict is keyed by GateCheck enum value
    (e.g. ``"text_preserve"``) → count of skipped passes.

    Use side-by-side with :func:`_stage7_violation_distribution`: a
    rising skip count for a check means the gate's coverage is
    silently shrinking even if violation counts stay flat.
    """
    counts: Counter[str] = Counter()
    for cand_results in results.values():
        for gr in cand_results:
            for check in gr.checks:
                if not getattr(check, "skipped", False):
                    continue
                # By contract a skipped check is also a passed check
                # (the skipped flag is an additional dimension on top
                # of pass/fail). Defensive belt-and-braces: only count
                # the passed-and-skipped intersection.
                if not check.passed:
                    continue
                counts[check.check.value] += 1
    return dict(counts)


#: Region kinds whose surrounding context carries load-bearing meaning —
#: a degraded neighbour risks SC 1.3.2 (Meaningful Sequence).
_COMPLEX_REGION_KINDS = frozenset({"table", "figure", "math"})


def _apply_h43_to_table_candidates(
    cands: dict[int, list[Any]],
    regions: list[Region],
) -> dict[int, list[Any]]:
    """Stage 6c — H43 id/headers enrichment on table candidates.

    Complex tables get deterministic ``<th id>``/``<td headers>``
    pairing (grid arithmetic, Plan 12 B1); the Stage-7 gate requires it.
    The id prefix carries the region index so ids stay unique after the
    assembler concatenates regions. ``H43Error`` propagates — a table
    candidate the enricher cannot parse must fail loudly here, not ship.
    """
    from .gates.table_h43 import apply_h43

    out: dict[int, list[Any]] = {}
    for idx, cand_list in cands.items():
        kind = regions[idx].kind if 0 <= idx < len(regions) else None
        if kind != "table":
            out[idx] = cand_list
            continue
        out[idx] = [
            dataclasses.replace(c, text=apply_h43(c.text, id_prefix=f"dart-t{idx}"))
            for c in cand_list
        ]
    return out


def _reading_order_at_risk_indices(
    top: dict[int, Any],
    regions: list[Region],
) -> list[int]:
    """Region indices whose Stage-7 wipe-out puts reading order at risk.

    A region is at risk when ALL its K candidates died at the gate
    (``top[idx] is None`` → the assembler emits the deterministic
    per-kind fallback) AND the region is, or is adjacent in reading
    order to, a complex region (table / figure / math). Plan 12 B3.
    """
    at_risk: list[int] = []
    n = len(regions)
    for idx in sorted(top):
        if top[idx] is not None:
            continue
        if not (0 <= idx < n):
            continue
        nearby = {regions[idx].kind} | {regions[j].kind for j in (idx - 1, idx + 1) if 0 <= j < n}
        if nearby & _COMPLEX_REGION_KINDS:
            at_risk.append(idx)
    return at_risk


def _stage10_axe_summary(gate_result: Any) -> dict[str, Any]:
    """Pull axe-only outcome from the document-level GateResult.

    Returns ``{"passed": bool, "violations": [{id, impact, nodes}, ...]}``
    so the caller can compute pass-rate / violation-distribution.
    """
    out: dict[str, Any] = {
        "passed": bool(gate_result.passed),
        "violations": [],
        "checks": [],
    }
    for check in gate_result.checks:
        out["checks"].append(
            {
                "check": check.check.value,
                "passed": bool(check.passed),
                "message": check.message,
            }
        )
        if check.check.value == "axe_wcag22aa":
            vlist = check.details.get("violations") if hasattr(check, "details") else None
            if vlist:
                out["violations"] = list(vlist)
    return out


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------


def run_full_cascade(
    pdf_path: Path,
    *,
    validator: HtmlValidator,
    max_regions_per_kind: int = 0,
    k: int = 2,
    runtime_mode: str = "mock",
    log: Callable[[str], None] = lambda _msg: None,
    enable_glm_ocr_stage: bool = False,
    glm_ocr_runtime: dict | None = None,
    return_html: bool = False,
) -> dict[str, Any]:
    """Run the full Stage 1..13 cascade against one PDF.

    The validator is owned by the caller (one Chromium instance shared
    across PDFs). ``max_regions_per_kind=0`` means no cap — use a small
    positive integer for the smoke harness.

    Stage 5b (``enable_glm_ocr_stage``) is OPT-IN. When enabled, runs
    GLM-OCR on table regions whose underlying TableCandidate the
    Structure council confirmed (see
    :func:`dart_semantic.glm_ocr_enrich.enrich_table_regions_with_glm_ocr`).
    With ``glm_ocr_runtime=None`` the stage runs in cache-only mode
    (cache hits get injected into payload; misses are silently skipped
    — caller should log a banner so the active mode is visible).

    ``return_html`` adds the assembled HTML string to the result dict
    under ``"html"``. The corpus eval driver leaves this off (it only
    wants ``html_length``, to keep per-PDF reports small); the v2
    pipeline entry point turns it on because the HTML *is* the product.

    Returns a per-PDF result dict containing:

    - ``pdf``                       : str of pdf_path
    - ``elapsed_total``             : float seconds
    - ``stages``                    : per-stage elapsed seconds (incl.
      ``stage5b`` when the stage ran)
    - ``region_kinds``              : Counter dict of capped region kinds
    - ``stage5b_glm_ocr_count``     : number of table regions enriched
      with GLM-OCR text (only when stage ran)
    - ``stage7_violations_under_mock``
    - ``stage7_skipped_under_mock`` : dict[check_name, count] of
      passed-but-skipped checks (no measurement against real source
      signal — e.g. text_preserve on gap-fill regions with no source
      text). Tracked separately from violations so the eval driver
      can flag silent gate-coverage holes.
    - ``stage7_n_regions``, ``stage7_n_candidates_in``,
      ``stage7_n_candidates_out``
    - ``stage10_axe``               : passed flag + violation list
    - ``heading_tree``              : list[(level, text)]
    - ``theta``                     : the ThetaReport, dataclass-as-dict
    - ``html_length``               : len(assembled.html)
    - ``html``                      : assembled HTML (only when
      ``return_html=True``)

    Raises any exception from the underlying pipeline; the caller
    (corpus driver / v2 entry point) is responsible for try/except per
    PDF. ``StageThirteenStubRequired`` from Stage 13 propagates
    unchanged — it is a deliberate loud failure, never degraded here.
    """
    stages: dict[str, float] = {}
    t_total = time.perf_counter()

    log(f"[cascade] running council on {pdf_path}")
    t = time.perf_counter()
    state, council_regions, feature_blocks = run_council(pdf_path)
    decisions = arbitrate(state, council_regions)
    structure_regions = build_structure_graph(
        state,
        feature_blocks,
        council_regions,
        decisions,
    )
    stages["stage1_5"] = time.perf_counter() - t

    # Stage 5b — optional GLM-OCR table enrichment. Runs ONLY on
    # table regions that the Structure council's table_region head
    # confirmed (cost-savings gate). Cache-only when glm_ocr_runtime
    # is None.
    stage5b_glm_ocr_count = 0
    if enable_glm_ocr_stage:
        log(
            f"[cascade] running Stage 5b (GLM-OCR table enrichment, "
            f"{'live' if glm_ocr_runtime is not None else 'cache-only'})"
        )
        t = time.perf_counter()
        from .extract_shared import extract_shared_cached
        from .glm_ocr_enrich import enrich_table_regions_with_glm_ocr

        shared = extract_shared_cached(pdf_path)
        before_with_text = sum(
            1 for r in structure_regions if r.kind == "table" and r.payload.get("glm_ocr_text")
        )
        structure_regions = enrich_table_regions_with_glm_ocr(
            structure_regions,
            council_regions,
            council_state=state,
            shared=shared,
            pdf_path=pdf_path,
            glm=glm_ocr_runtime,
        )
        after_with_text = sum(
            1 for r in structure_regions if r.kind == "table" and r.payload.get("glm_ocr_text")
        )
        stage5b_glm_ocr_count = max(0, after_with_text - before_with_text)
        stages["stage5b"] = time.perf_counter() - t
        log(
            f"[cascade] Stage 5b enriched {stage5b_glm_ocr_count} table region(s) with GLM-OCR text"
        )

    # Stage 5c — render figure Region bboxes to PNG bytes (Plans/09 §1).
    # Auto no-op when no figure Regions are present. Without pixels, Stage 6's
    # captioner has nothing to look at, so unlike Stage 5b this isn't opt-in.
    n_figs_before = sum(1 for r in structure_regions if r.kind == "figure")
    if n_figs_before:
        log(f"[cascade] running Stage 5c (figure-bbox PNG rendering, n_figures={n_figs_before})")
        t = time.perf_counter()
        from .image_extract import render_figure_regions_to_bytes

        structure_regions = render_figure_regions_to_bytes(
            structure_regions,
            feature_blocks,
            pdf_path,
        )
        stages["stage5c"] = time.perf_counter() - t

    capped = _cap_regions_per_kind(
        structure_regions,
        cap=max_regions_per_kind,
    )
    region_kinds = dict(Counter(r.kind for r in capped))
    log(f"[cascade] capped region count by kind: {region_kinds}")

    # Stage 6b — figure captioner (SmolVLM2, Plans/09 §2). Runs ONCE before
    # the lanes since alt_text is lane-independent; reads payload['image_png_bytes']
    # from Stage 5c and attaches alt_text + extended_description for the
    # assembler. Auto no-op when no figure Regions are present.
    n_figs_capped = sum(1 for r in capped if r.kind == "figure")
    if n_figs_capped:
        log(f"[cascade] running Stage 6b (figure captioner, n_figures={n_figs_capped})")
        t = time.perf_counter()
        from .figure_captioner import caption_figure_regions

        capped = caption_figure_regions(capped)
        stages["stage6b"] = time.perf_counter() - t

    # Evict the council BERTs from the GPU before Stage 6 loads the Qwen GGUF.
    # Stages 6-12 don't use the shared backbone (Qwen is llama-cpp, theta is its
    # own DeBERTa, the soft rerankers are heuristic), so this is safe and frees
    # ~2-3 GB — without it, large PDFs OOM on GGUF load on the 8 GB card even in
    # an isolated process ("one model resident at a time" across BERT->Qwen).
    from .council.base import release_council_gpu

    release_council_gpu()

    # ------------------------------------------------------------------
    # Stages 6-12 are encapsulated so the offline-retry orchestrator can
    # re-run the inner pipeline against ``lane="offline"``. Stage outputs
    # that the LAST lane produced are captured into ``lane_outputs`` so
    # the per-PDF JSON report describes the lane that ultimately won
    # (matched by the FINAL ThetaReport's ``lane`` field).
    # ------------------------------------------------------------------
    lane_outputs: dict[str, dict[str, Any]] = {}

    def _run_inner(lane: str) -> Any:
        """Stages 6-12 for a given lane. Returns a fresh ThetaReport.

        Side-effect: populates ``lane_outputs[lane]`` with the artifacts
        downstream code needs (axe summary, assembled doc, etc.).
        Per-stage timings are recorded with a ``_offline`` suffix on the
        offline pass so totals don't collide.
        """
        suffix = "" if lane == "fast" else "_offline"

        log(
            f"[cascade] [{lane}] running Stage 6 ({runtime_mode} runtime, k={k}) "
            f"on {len(capped)} regions"
        )
        t_inner = time.perf_counter()
        cands = run_qwen_specialists(
            capped,
            feature_blocks,
            k=k,
            runtime_mode=runtime_mode,
            lane=lane,  # type: ignore[arg-type]
        )
        stages[f"stage6{suffix}"] = time.perf_counter() - t_inner

        n_in = sum(len(v) for v in cands.values())

        # Stage 6c — deterministic H43 enrichment (Plan 12 B1): complex
        # tables (multi header rows / dual axis / spanned <th>) get
        # id/headers association generated from grid arithmetic BEFORE
        # the gate, which now requires it on complex tables. Simple
        # tables pass through byte-identical.
        cands = _apply_h43_to_table_candidates(cands, capped)

        log(f"[cascade] [{lane}] running Stage 7 (per-region hard gate)")
        t_inner = time.perf_counter()
        survs, res = gate_per_region(
            cands,
            capped,
            feature_blocks,
            validator=validator,
        )
        stages[f"stage7{suffix}"] = time.perf_counter() - t_inner
        n_out = sum(len(v) for v in survs.values())
        s7_dist = _stage7_violation_distribution(res)
        s7_skipped = _stage7_skip_distribution(res)

        log(f"[cascade] [{lane}] running Stage 8 (per-region soft reranker)")
        t_inner = time.perf_counter()
        top = rerank_per_region(survs, capped, feature_blocks)
        stages[f"stage8{suffix}"] = time.perf_counter() - t_inner

        log(f"[cascade] [{lane}] running Stage 9 (assembler, {runtime_mode} runtime)")
        t_inner = time.perf_counter()
        asm = assemble_document(
            top,
            capped,
            feature_blocks,
            council_state=state,
            runtime_mode=runtime_mode,
            config=AssemblerConfig(),
            validator=validator,
        )
        stages[f"stage9{suffix}"] = time.perf_counter() - t_inner

        log(f"[cascade] [{lane}] running Stage 10 (document-level hard gate)")
        t_inner = time.perf_counter()
        gr = gate_document(asm, validator=validator)
        stages[f"stage10{suffix}"] = time.perf_counter() - t_inner
        s10_axe = _stage10_axe_summary(gr)

        log(f"[cascade] [{lane}] running Stage 11 (document-level soft reranker)")
        t_inner = time.perf_counter()
        sc = score_document(asm)
        stages[f"stage11{suffix}"] = time.perf_counter() - t_inner

        log(f"[cascade] [{lane}] running Stage 12 (ThetaEvaluator)")
        t_inner = time.perf_counter()
        wcag = "passed" if gr.passed else "failed"
        ro_at_risk = _reading_order_at_risk_indices(top, list(capped))
        rep = evaluate(
            asm,
            feature_blocks=feature_blocks,
            regions=capped,
            wcag_status=wcag,
            lane=lane,  # type: ignore[arg-type]
            stage11_scored=sc,
            reading_order_at_risk=ro_at_risk,
        )
        stages[f"stage12{suffix}"] = time.perf_counter() - t_inner

        lane_outputs[lane] = {
            "n_cand_in": n_in,
            "n_cand_out": n_out,
            "stage7_dist": s7_dist,
            "stage7_skipped": s7_skipped,
            "stage7_results": res,
            "n_regions_no_survivor": sum(1 for v in top.values() if v is None),
            "reading_order_at_risk": ro_at_risk,
            "assembled": asm,
            "gate_result": gr,
            "stage10_axe": s10_axe,
            "wcag_status": wcag,
        }
        return rep

    fast_report = _run_inner("fast")

    # Stage 13 — orchestrate the offline retry first, THEN stamp.
    offline_retry_fired = False

    def _run_lane(lane: str) -> Any:
        nonlocal offline_retry_fired
        if lane == "offline":
            offline_retry_fired = True
        return _run_inner(lane)

    log("[cascade] running Stage 13 (offline retry orchestration)")
    t = time.perf_counter()
    final_report = maybe_offline_retry(fast_report, run_lane=_run_lane)
    report = decide_exit(final_report)
    stages["stage13"] = time.perf_counter() - t

    # Pick the lane outputs that match the FINAL report (the report's
    # ``lane`` field is set by the lane that actually produced it; the
    # reconcile rules in offline_retry pick a winner).
    lane_used = report.lane
    chosen = lane_outputs.get(lane_used) or lane_outputs["fast"]
    n_cand_in = chosen["n_cand_in"]
    n_cand_out = chosen["n_cand_out"]
    stage7_dist = chosen["stage7_dist"]
    stage7_skipped = chosen["stage7_skipped"]
    assembled = chosen["assembled"]
    gate_result = chosen["gate_result"]
    stage10_axe = chosen["stage10_axe"]
    wcag_status = chosen["wcag_status"]

    elapsed_total = time.perf_counter() - t_total

    theta_payload = dataclasses.asdict(report)

    # Council per-head signal coverage — observability metadata so a
    # partial-failure run (head load fails, OOM mid-batch, signals
    # dropped) can't masquerade as healthy. See CouncilState.
    council_signal_coverage = dict(getattr(state, "signal_coverage", {}) or {})

    # Per-document conformance audit (Plan 12 A1) — the verifiability
    # artifact. Built for EVERY exit mode from the winning lane's gate
    # state; scripts/pdf_to_html.py persists it next to the product.
    from .conformance_audit import build_conformance_audit
    from .gates.wcag_coverage import coverage_map as _coverage_map

    conformance_audit = build_conformance_audit(
        pdf_path=str(pdf_path),
        runtime_mode=runtime_mode,
        k=k,
        lane_used=lane_used,
        offline_retry_fired=offline_retry_fired,
        exit_action=report.action,
        region_gate_results=chosen["stage7_results"],
        regions=list(capped),
        stage7_skip_counts=stage7_skipped,
        stage7_violation_counts=stage7_dist,
        n_candidates_in=n_cand_in,
        n_candidates_out=n_cand_out,
        n_regions_no_survivor=chosen["n_regions_no_survivor"],
        reading_order_at_risk=chosen["reading_order_at_risk"],
        doc_gate_result=gate_result,
        theta_report=report,
        assembled=assembled,
        wcag_coverage=_coverage_map(),
    )

    result: dict[str, Any] = {
        "pdf": str(pdf_path),
        "elapsed_total": elapsed_total,
        "stages": stages,
        "region_kinds": region_kinds,
        "n_regions_capped": len(capped),
        "n_regions_total": len(structure_regions),
        "stage5b_enabled": bool(enable_glm_ocr_stage),
        "stage5b_glm_ocr_count": stage5b_glm_ocr_count,
        "stage7_n_candidates_in": n_cand_in,
        "stage7_n_candidates_out": n_cand_out,
        "stage7_violations_under_mock": stage7_dist,
        "stage7_skipped_under_mock": stage7_skipped,
        "stage10_pass_under_mock": bool(gate_result.passed),
        "stage10_axe_under_mock": stage10_axe,
        "wcag_status_under_mock": wcag_status,
        "heading_tree": [list(t) for t in assembled.heading_tree],
        "landmarks": dict(assembled.landmarks),
        "html_length": len(assembled.html),
        "theta": theta_payload,
        "council_signal_coverage": council_signal_coverage,
        "conformance_audit": conformance_audit,
        # Stage 13 retry-orchestrator metadata. ``lane_used`` is the
        # lane that produced the FINAL report (see offline_retry.py
        # reconciliation rules). ``offline_retry_fired`` is True iff
        # the orchestrator decided to dispatch the offline lane (it
        # may have done so and still kept the fast report).
        "lane_used": lane_used,
        "offline_retry_fired": offline_retry_fired,
        "offline_retry_won": offline_retry_fired and lane_used == "offline",
    }
    if return_html:
        result["html"] = assembled.html
    return result


# ---------------------------------------------------------------------------
# v2 pipeline entry point
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PipelineV2Result:
    """Result of a ``mode="v2"`` pipeline run — the assembled product
    plus the Stage 13 exit decision.

    Wraps the certified (or flagged / non-certified) HTML and the final
    :class:`~dart_semantic.theta.types.ThetaReport`. ``cascade`` carries
    the full per-stage telemetry dict from :func:`run_full_cascade` for
    callers that want stage timings, gate distributions, or coverage.
    """

    pdf: str
    html: str
    wcag_status: str  # "passed" | "failed"
    exit_action: str  # ConfidenceAction enum value
    theta_score: float | None
    flags: list[str]  # ThetaFlag enum values
    lane_used: str
    theta_report: dict  # ThetaReport, dataclass-as-dict
    cascade: dict  # full run_full_cascade result dict


def _enum_value(obj: Any) -> Any:
    """Return ``obj.value`` for an enum member, else ``obj`` unchanged."""
    return obj.value if hasattr(obj, "value") else obj


def run_pipeline_v2(
    pdf_path: Path | str,
    *,
    config: V2Config = DEFAULT_V2_CONFIG,
    k: int = 4,
    max_regions_per_kind: int = 0,
    runtime_mode: str | None = None,
    enable_glm_ocr_stage: bool = False,
    log: Callable[[str], None] = lambda _msg: None,
) -> PipelineV2Result:
    """Run the v2 pipeline (Stage 1..13) end-to-end on one PDF.

    Owns the :class:`HtmlValidator` (Chromium) lifecycle for the run —
    callers that already hold a validator and want to amortize it across
    many PDFs should call :func:`run_full_cascade` directly.

    Parameters
    ----------
    pdf_path
        PDF to process.
    config
        :class:`V2Config` selecting which adapters are loaded. When no
        Qwen specialist adapter is configured, ``runtime_mode`` defaults
        to ``"mock"`` (the LoRA adapters don't exist on disk yet); once
        adapters are registered it defaults to ``"real"``.
    k
        Candidates per region from Stage 6. Architecture default for the
        fast lane is K=4.
    max_regions_per_kind
        Region cap per kind (0 = no cap). Smoke harnesses pass a small
        positive int; production passes 0.
    runtime_mode
        Force ``"mock"`` or ``"real"``. ``None`` (default) derives it
        from ``config``.
    enable_glm_ocr_stage
        Opt-in Stage 5b GLM-OCR table enrichment (cache-only here).
    log
        Optional line logger for cascade progress banners.

    Returns
    -------
    PipelineV2Result
        The assembled HTML and the Stage 13 exit decision.

    Raises
    ------
    StageThirteenStubRequired
        When the Stage 13 decision table routes to an offline-Qwen lane
        that v1 doesn't implement. This is a deliberate loud failure —
        it is NOT degraded to a flagged result here (see
        ``feedback_no_silent_fallbacks``).
    """
    pdf_path = Path(pdf_path)
    if runtime_mode is None:
        runtime_mode = "real" if config.is_qwen_specialist_loaded() else "mock"

    with HtmlValidator() as validator:
        result = run_full_cascade(
            pdf_path,
            validator=validator,
            max_regions_per_kind=max_regions_per_kind,
            k=k,
            runtime_mode=runtime_mode,
            enable_glm_ocr_stage=enable_glm_ocr_stage,
            return_html=True,
            log=log,
        )

    theta = result["theta"]
    flags = [_enum_value(f) for f in (theta.get("flags") or [])]
    return PipelineV2Result(
        pdf=result["pdf"],
        html=result["html"],
        wcag_status=result["wcag_status_under_mock"],
        exit_action=_enum_value(theta.get("action")),
        theta_score=theta.get("theta_score"),
        flags=flags,
        lane_used=result["lane_used"],
        theta_report=theta,
        cascade=result,
    )


__all__ = [
    "run_full_cascade",
    "run_pipeline_v2",
    "PipelineV2Result",
]
