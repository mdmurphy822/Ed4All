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
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .assembler import AssemblerConfig, assemble_document
from .council.cross_reranker import arbitrate
from .council.orchestrator import run_council
from .gates import gate_document, gate_per_region, rerank_per_region
from .qwen_specialists.runner import run_qwen_specialists
from .qwen_specialists.reviewer import resolve_structure_review_mode
from .qwen_specialists.runtime import any_phase_provider_is_endpoint
from .reading_order import resolve_deploy_profile
from .soft_reranker import score_document
from .stop_seam import check_cascade_stop
from .structure_graph import Region, build_structure_graph, stamp_role_distributions
from .theta import apply_repair_stats, decide_exit, evaluate, maybe_offline_retry
from .v2_config import DEFAULT_V2_CONFIG, V2Config
from .validate import HtmlValidator


# ---------------------------------------------------------------------------
# Deterministic GPU-lifecycle lease (ED4ALL_GPU_LIFECYCLE, default ON)
# ---------------------------------------------------------------------------


def _gpu_lifecycle_release(
    *, ollama: bool = False, torch: bool = False, stage: str = ""
) -> None:
    """Fire a gated, fail-soft GPU-lifecycle release at a cascade stage seam.

    Deterministic lease semantics (owner directive): a GPU model stays resident
    for its stage's batch and releases the card at the SEAM to the next stage
    (post-Stage-5e ollama hand-off before the Stage-6 GGUF; post-captioner /
    post-Stage-6 / post-theta torch; post second-pass+ocr_repair ollama). Uses
    the SELF-CONTAINED SemantiK twin (``dart_semantic.gpu_lifecycle``) because
    the out-of-process cascade bridge cannot import Ed4All's ``lib/``.

    No-op (ZERO release calls) when ``ED4ALL_GPU_LIFECYCLE`` is off — the
    flag-off cascade path is byte-identical. Wraps the already-fail-soft twin in
    a final belt so a seam release can NEVER crash a cascade stage. Every seam
    is idempotent + lazy-reload-safe (ollama auto-loads; the Stage-6 AdapterSwap
    reloads; theta reloads on the next evaluate), so ``maybe_offline_retry``'s
    re-entry of ``_run_inner`` is safe.
    """
    try:
        from . import gpu_lifecycle

        if not gpu_lifecycle.resolve_gpu_lifecycle_mode():
            return
        if ollama:
            gpu_lifecycle.release_ollama_models(stage=stage or None)
        if torch:
            gpu_lifecycle.release_torch(stage=stage or None)
    except Exception:  # noqa: BLE001 — a seam release must never crash the cascade
        pass


# ---------------------------------------------------------------------------
# Part F — figure-detection flags
# ---------------------------------------------------------------------------

_DETECT_FIGURES_ENV = "SEMANTIK_DETECT_FIGURES"
_FIGURE_CAPTION_ENV = "SEMANTIK_FIGURE_CAPTION"
_FIG_TRUTHY = {"1", "true", "yes", "on"}
_FIG_FALSEY = {"0", "false", "no", "off"}


def resolve_detect_figures() -> bool:
    """Whether the Part F figure-detection path (Stages A-D) runs.

    Reads ``SEMANTIK_DETECT_FIGURES`` OR the bundled ``SEMANTIK_DEPLOY_PROFILE``
    (W7.1 — the deploy profile turns figure detection AND column reading-order
    on together). DEFAULT OFF (byte-stable). Parse-with-fallback: truthy
    (``1``/``true``/``yes``/``on``) on EITHER flag → on; unset / falsey /
    garbage → off. Resolver of record; the extract-side mirror
    ``extract_shared._detect_figures_enabled`` reads the same envs (extract
    runs inside the council orchestrator, off the cascade's thread). Mirrors
    ``ED4ALL_BLOCK_ANATOMY``'s default-off posture.
    """
    if os.environ.get(_DETECT_FIGURES_ENV, "").strip().lower() in _FIG_TRUTHY:
        return True
    return resolve_deploy_profile()


def resolve_figure_caption_mode() -> str:
    """Tri-state caption mode: ``"on"`` / ``"off"`` / ``"auto"`` (default).

    Reads ``SEMANTIK_FIGURE_CAPTION``. ``auto`` (unset / "auto" / garbage)
    captions when CUDA is present AND the flag is not explicitly off; an
    explicit truthy forces on; an explicit falsey defers captioning (the
    figure ships the honest type-level ``"Figure."`` alt). Only consulted
    when ``resolve_detect_figures`` is on.
    """
    raw = os.environ.get(_FIGURE_CAPTION_ENV, "").strip().lower()
    if raw in _FIG_TRUTHY:
        return "on"
    if raw in _FIG_FALSEY:
        return "off"
    return "auto"


def _figure_captioning_active() -> bool:
    """Resolve the tri-state caption mode to a concrete on/off decision."""
    mode = resolve_figure_caption_mode()
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto — caption only when a CUDA device is available (the SmolVLM2
    # captioner is too slow CPU-only; defer to the type-level alt).
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — no torch / no cuda → defer
        return False


# ---------------------------------------------------------------------------
# Stage-output aggregation helpers
# ---------------------------------------------------------------------------


def _region_min_fb(region: Region) -> int | None:
    """The region's STABLE FB-derived key = ``min(feature_block_indices)``.

    Unique per surviving region (R-PART partitions FBs, never shared), so it is
    a clean bijection that survives an N->1 Stage-5e region merge (a merged
    region inherits the anchor LABEL's min-FB). ``None`` for an FB-less region
    (pathological — used as the join key only when present)."""
    fb = getattr(region, "feature_block_indices", ()) or ()
    return min(fb) if fb else None


def _cap_regions_per_kind(
    regions: list[Region],
    *,
    cap: int,
    stage5d_metadata_drop_ids: frozenset[int] | None = None,
) -> list[Region]:
    """Cap the region list to ``cap`` survivors PER KIND (lossy ``continue``).

    ``stage5d_metadata_drop_ids`` (C2 cap-exemption): the set of STABLE
    ``min(feature_block_indices)`` KEYS (Phase 5 re-key — previously POSITIONAL
    region indices) of the regions the Stage-5d reviewer NEWLY re-tagged to
    ``metadata_drop`` this run. A re-tagged-this-stage ``metadata_drop`` region
    is exempt from counting against ANY bucket's running cap — it emits empty
    HTML and is filtered out of ``body_html`` downstream, so counting it against
    a content bucket would needlessly evict real content (a phantom heading
    re-roled to ``paragraph``/``metadata_drop`` must NOT push a real paragraph
    out of the cap). The region still SURVIVES into the output list (coverage
    invariant: every FB stays owned); it just does not consume a cap slot.

    Keying by min-FB (not by position) means the exemption survives a Stage-5e
    N->1 region merge that shifts every downstream position — the misaligned
    positional exemption could otherwise let the cap evict a real surviving
    region and trip the fail-closed cap-safety revert (silently discarding the
    whole regroup). Byte-stable when no merge fires: with positions unchanged
    each region's min-FB equals its old positional index, so the SET of exempt
    regions is identical to the historical per-kind cap.

    When ``stage5d_metadata_drop_ids`` is None/empty (flag OFF, or no drops),
    this is byte-identical to the historical per-kind cap.
    """
    if cap <= 0:
        return list(regions)
    exempt = stage5d_metadata_drop_ids or frozenset()
    counts: Counter[str] = Counter()
    out: list[Region] = []
    for r in regions:
        # A Stage-5d metadata_drop re-tag is cap-exempt: keep it (empty-emit,
        # dropped from body_html) but never let it consume a bucket slot.
        # Resolve the exemption by the region's STABLE min-FB key (Phase 5).
        min_fb = _region_min_fb(r)
        if min_fb is not None and min_fb in exempt:
            out.append(r)
            continue
        if counts[r.kind] >= cap:
            continue
        out.append(r)
        counts[r.kind] += 1
    return out


def _surviving_fb_indices(regions: list[Region]) -> Counter[int]:
    """Multiset of FeatureBlock indices owned by the given (surviving) regions.

    A re-role-induced cap eviction silently drops a region AND its
    FeatureBlocks; comparing this multiset for the flag-ON survivors against
    the flag-OFF baseline is the C2 / R12 FB-survival check.
    """
    owned: Counter[int] = Counter()
    for r in regions:
        for idx in (getattr(r, "feature_block_indices", ()) or ()):
            owned[idx] += 1
    return owned


def _assert_fb_survival_through_cap(
    baseline_capped: list[Region],
    reviewed_capped: list[Region],
) -> None:
    """C2 / R12 — fail closed if a Stage-5d re-role evicted real content.

    The multiset of FeatureBlock indices that survive the cap with the
    Stage-5d corrections applied MUST be a superset of (here: equal to) the
    multiset that survives the cap on the flag-OFF baseline. A real-content
    FB that the baseline kept but the reviewed run dropped is a re-role-
    induced eviction — the cap pushing a real paragraph out because N
    phantom headings re-roled into its bucket. We fail closed (caller
    reverts the Stage-5d corrections for the run) rather than silently
    shipping a document with lost content.

    Raises :class:`CapSafetyError` on any baseline FB missing from the
    reviewed survivors.
    """
    baseline_fbs = _surviving_fb_indices(baseline_capped)
    reviewed_fbs = _surviving_fb_indices(reviewed_capped)
    missing = baseline_fbs - reviewed_fbs
    if missing:
        raise CapSafetyError(
            "structure-review cap-safety FAILED: "
            f"{sum(missing.values())} FeatureBlock(s) {sorted(missing)!r} "
            "survived the per-kind cap at baseline but were EVICTED after the "
            "Stage-5d re-role (a phantom-heading re-role pushed real content "
            "past the cap) — failing closed rather than dropping content."
        )


class CapSafetyError(RuntimeError):
    """Raised when a Stage-5d re-role evicts real content at the per-kind cap.

    Caught by ``run_full_cascade`` to FAIL CLOSED: the Stage-5d corrections
    are reverted to the flag-OFF region list for the run (C2 / R12)."""


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
# P3a — per-region provenance distillation (SemantiK migration §3.3a/§3.5).
#
# ``run_full_cascade`` computes everything the Ed4All adapter IR needs
# (regions = ``capped``, ``feature_blocks``, per-region Stage-7 gate
# results) but historically dropped it from the result. This helper
# distills a STABLE, in-memory per-region provenance list in DOCUMENT
# order so the in-process seam (``lib/semantik/cascade_ir.py``) can build
# the adapter's chapters IR without re-running models. No cascade LOGIC
# changes — capture-and-expose only. Every field guards for absence so a
# mock run (empty payloads / partial feature blocks) never raises.
# ---------------------------------------------------------------------------


def _fb_page(feature_blocks: list[Any], fb_idx: int) -> int | None:
    """1-indexed physical page of a feature block, or ``None`` if absent."""
    if not (0 <= fb_idx < len(feature_blocks)):
        return None
    fb = feature_blocks[fb_idx]
    raw = getattr(fb, "raw", None)
    page = getattr(raw, "page", None)
    try:
        p = int(page)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def _fb_text(feature_blocks: list[Any], fb_idx: int) -> str:
    """Deterministic extracted text of a feature block (``""`` if absent)."""
    if not (0 <= fb_idx < len(feature_blocks)):
        return ""
    fb = feature_blocks[fb_idx]
    raw = getattr(fb, "raw", None)
    return (getattr(raw, "text", None) or "") if raw is not None else ""


def _region_wcag_status(stage7_results: dict[int, list[Any]], region_index: int) -> str | None:
    """Per-region WCAG verdict distilled from the Stage-7 candidate gate.

    ``"passed"`` iff at least one candidate survived (matches the Stage-7
    survivor semantics in :func:`build_region_gate_log`); ``"failed"`` when
    every candidate died; ``None`` when the region was never gated (no
    entry — e.g. a passthrough region).
    """
    cands = stage7_results.get(region_index)
    if cands is None:
        return None
    if any(getattr(gr, "passed", False) for gr in cands):
        return "passed"
    return "failed"


def _review_by_region_index(
    verdicts: list[Any] | None,
    review_regions: list[Region] | None = None,
) -> dict[int, dict[str, Any]]:
    """Index Stage-5d ReviewVerdicts by their join key, for
    ``_build_region_provenance``.

    Returns a terse per-region ``review`` payload (``corrected_from`` /
    ``corrected_to`` / ``reason_code`` / ``reverted`` / ``note``). Only
    NON-``ok`` (corrected or reverted) verdicts are surfaced — an ``ok``
    no-op carries no audit signal and would only bloat the provenance.
    Empty dict when ``verdicts`` is None (flag OFF) so the provenance dict
    stays byte-stable.

    Join key (Phase 5): when ``review_regions`` (the pre-Stage-5e region list
    the verdict ``block_id``s index into) is provided, the dict is keyed by the
    STABLE ``min(feature_block_indices)`` of ``review_regions[block_id]`` so the
    join survives a Stage-5e N->1 region merge that shifts every post-5e
    position. When ``review_regions`` is None (legacy / direct callers) the dict
    is keyed by ``block_id`` (the pre-5e positional region index) EXACTLY as
    before — byte-identical for callers that don't opt into the re-key.
    """
    out: dict[int, dict[str, Any]] = {}
    n_review = len(review_regions) if review_regions is not None else 0
    for v in (verdicts or ()):
        kind_before = getattr(v, "kind_before", None)
        kind_after = getattr(v, "kind_after", None)
        level_before = getattr(v, "level_before", None)
        level_after = getattr(v, "level_after", None)
        reverted = bool(getattr(v, "reverted_for_invariant", False))
        endpoint_degraded = bool(
            getattr(v, "reverted_for_endpoint_failure", False)
        )
        verdict_label = getattr(v, "verdict", "ok")
        changed = (kind_before != kind_after) or (level_before != level_after)
        if not changed and not reverted and not endpoint_degraded:
            # Pure no-op ``ok`` — no audit signal to carry. An endpoint-
            # degraded cluster IS surfaced (it must not silently look
            # "reviewed" — it was never reviewed; the endpoint was down).
            continue
        bid = getattr(v, "block_id", None)
        if not isinstance(bid, int):
            continue
        # Phase 5: re-key the verdict by the pre-5e region's stable min-FB so
        # the join survives a Stage-5e N->1 merge. Falls back to the positional
        # block_id (byte-identical legacy path) when no review_regions / the
        # block has no resolvable min-FB.
        key = bid
        if review_regions is not None and 0 <= bid < n_review:
            min_fb = _region_min_fb(review_regions[bid])
            if min_fb is not None:
                key = min_fb
        out[key] = {
            "corrected_from": kind_before,
            "corrected_to": kind_after,
            "level_from": level_before,
            "level_to": level_after,
            "reason_code": verdict_label,
            "reverted": reverted,
            "reverted_for_endpoint_failure": endpoint_degraded,
            "note": getattr(v, "review_note", "") or "",
        }
    return out


def _write_figure_sidecars(
    regions: list[Region],
    pdf_path: Path,
    *,
    log: Callable[[str], None] = lambda _msg: None,
) -> list[Region]:
    """Stage F — write each figure region's PNG to a deterministic sidecar
    and stamp ``payload["image_src"]``; strip the raw bytes.

    Sidecar layout: ``{pdf_stem}_figures/fig-{first_fb_index}.png`` next to
    the source PDF. ``image_src`` is the RELATIVE path
    ``./{pdf_stem}_figures/fig-N.png`` (the emitters resolve it against the
    written HTML location). The raw ``image_png_bytes`` is removed from the
    payload after the write so it never reaches the JSON bridge (only
    ``image_src`` + ``figure_alt`` travel). Per-region fail-soft: a write
    failure leaves that figure with no ``image_src`` (its emitter falls
    back to the text-only ``<figure>``) but never aborts the document.

    The Region dataclass is frozen, so each touched region is replaced.
    """
    from dataclasses import replace as _dc_replace

    stem = pdf_path.stem
    figures_dirname = f"{stem}_figures"
    figures_dir = pdf_path.parent / figures_dirname
    out: list[Region] = []
    n_written = 0
    dir_made = False
    for region in regions:
        if getattr(region, "kind", None) != "figure":
            out.append(region)
            continue
        payload = dict(region.payload or {})
        png_bytes = payload.pop("image_png_bytes", None)
        if not png_bytes:
            # No pixels (deferred render / failed bbox) — text-only figure.
            out.append(_dc_replace(region, payload=payload))
            continue
        fb_indices = getattr(region, "feature_block_indices", ()) or ()
        first_fb = min(fb_indices) if fb_indices else 0
        fname = f"fig-{first_fb}.png"
        try:
            if not dir_made:
                figures_dir.mkdir(parents=True, exist_ok=True)
                dir_made = True
            (figures_dir / fname).write_bytes(png_bytes)
            payload["image_src"] = f"./{figures_dirname}/{fname}"
            n_written += 1
        except Exception as exc:  # noqa: BLE001 — degrade one figure
            log(f"[cascade] Stage F sidecar write failed for fb={first_fb}: {exc}")
            # image_png_bytes already popped; no src → text-only figure.
        out.append(_dc_replace(region, payload=payload))
    log(f"[cascade] Stage F wrote {n_written} figure sidecar PNG(s) to {figures_dir}")
    return out


def _build_region_provenance(
    region_order: list[int],
    regions: list[Region],
    feature_blocks: list[Any],
    stage7_results: dict[int, list[Any]],
    review_verdicts: list[Any] | None = None,
    review_regions: list[Region] | None = None,
    ocr_repair_edits: dict[int, dict[str, Any]] | None = None,
    containment_tree: Any | None = None,
) -> list[dict[str, Any]]:
    """Distill one provenance dict per region in DOCUMENT (emission) order.

    ``region_order`` is ``AssembledDoc.region_provenance`` — for each
    emitted block, the index into ``regions`` (= ``capped``). The assembler
    emits regions in reading order, so iterating it yields document order.

    Each dict carries the §3.3a/§3.5 fields the adapter IR consumes:

    - ``region_index``        : index into ``regions`` (= ``capped``)
    - ``region_kind``         : the typed RegionKind
    - ``role``                : Semantic ``doc_role`` payload label (or kind)
    - ``confidence``          : per-region cascade confidence (payload)
    - ``wcag_status``         : per-region Stage-7 verdict (passed/failed/None)
    - ``first_raw_block_index``: §3.3a determinism key — the SMALLEST raw
      feature-block index the region claims (stable under merge/split)
    - ``pages``               : sorted 1-indexed physical pages the region
      spans (from each claimed FB's ``raw.page``)
    - ``heading_text``        : heading/figure label (payload ``text``)
    - ``level``               : heading level hint (payload ``level_hint``)
    - ``figure_alt``          : Stage-6b caption for figure regions
    - ``raw_text``            : concatenated deterministic extracted text
      (hash basis for content-hash sids; never post-model HTML)
    - ``pedagogy_class``      : OPTIONAL — the semantic class hint
      (``"pedagogy-example"`` / ``"pedagogy-solution"`` / …) stamped by the
      deterministic structure-correction pass on a demoted pedagogical
      paragraph; present only when set (byte-stable to baseline when absent)
    """
    provenance: list[dict[str, Any]] = []
    n = len(regions)
    # Phase 5: when ``review_regions`` (the pre-5e region list the verdict
    # block_ids index into) is supplied, the verdict join is keyed by stable
    # min-FB so it survives a Stage-5e N->1 merge; otherwise the legacy
    # positional (region_index == block_id) join is used (byte-identical).
    review_index = _review_by_region_index(review_verdicts, review_regions)
    for region_index in region_order:
        if not (0 <= region_index < n):
            # Defensive: a provenance index outside the region list is a
            # bug upstream; skip rather than raise so a mock run survives.
            continue
        region = regions[region_index]
        fb_indices = list(getattr(region, "feature_block_indices", ()) or ())
        payload = getattr(region, "payload", {}) or {}

        first_raw = min(fb_indices) if fb_indices else region_index
        pages = sorted(
            {p for fb in fb_indices if (p := _fb_page(feature_blocks, fb)) is not None}
        )
        raw_text = " ".join(
            t for fb in sorted(fb_indices) if (t := _fb_text(feature_blocks, fb).strip())
        )

        kind = getattr(region, "kind", "paragraph")
        role = payload.get("doc_role") or kind
        confidence = payload.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        heading_text = payload.get("text") if kind in {"heading", "figure"} else None
        figure_alt = (payload.get("alt_text") or None) if kind == "figure" else None
        # Part F — relative ``<img src>`` to the figure sidecar PNG. Set by
        # the Stage-6b/F sidecar-write pass on the region payload; absent
        # (None) for every non-figure region and byte-stable when the
        # figure path is off.
        image_src = (payload.get("image_src") or None) if kind == "figure" else None

        # Resolve the figure/table CAPTION text from ``caption_fb_index`` (the
        # structure-graph caption-neighbor pass stamps the index of the
        # "Figure N:" / "Table N:" FB on the payload, NOT the text). Carrying
        # the resolved text here is the only place the cascade_ir adapter — which
        # consumes ONLY the distilled provenance list, never the rich assembled
        # HTML — can read a figure's caption (a synthetic image FB has empty
        # ``raw_text``, so without this a figure provenance entry has no
        # accessible name and the adapter emitted an empty ``<figure></figure>``).
        caption_text: str | None = None
        if kind in {"figure", "table"}:
            cap_idx = payload.get("caption_fb_index")
            if isinstance(cap_idx, int):
                cap = _fb_text(feature_blocks, cap_idx).strip()
                if cap:
                    caption_text = cap

        entry: dict[str, Any] = {
            "region_index": region_index,
            "region_kind": kind,
            "role": role,
            "confidence": confidence,
            "wcag_status": _region_wcag_status(stage7_results, region_index),
            "first_raw_block_index": int(first_raw),
            "pages": pages,
            "heading_text": heading_text,
            "level": payload.get("level_hint"),
            "figure_alt": figure_alt,
            "raw_text": raw_text,
        }
        # OPTIONAL ``image_src`` key (additive). Present only for a figure
        # region with a written sidecar PNG — byte-stable to baseline (the
        # key is simply absent) for every non-figure region and for runs
        # with the figure path off.
        if image_src is not None:
            entry["image_src"] = image_src
        # OPTIONAL P2 ``vlm_corroborated`` key (additive audit signal). Stamped
        # on the payload ONLY when a whole-block VLM heading hint corroborated a
        # heading candidate that had already cleared every deterministic gate
        # (structure_graph Pass-2). Absent (byte-stable to baseline) for every
        # region when the VLM struct-hints flags are off. The boosted
        # ``confidence`` above rides the existing key — no new confidence field.
        if payload.get("vlm_corroborated"):
            entry["vlm_corroborated"] = True
        # OPTIONAL ``role_top_k`` key (additive, ITEM6). Present only when the
        # Stage-5 stamp ran (a real council state); byte-stable to baseline
        # (key simply absent) for mock/legacy runs. Deliberately NOT lifting
        # ``pedagogical_top_k`` — no named wire consumer; it stays on
        # Region.provenance for in-run consumers only.
        role_top_k = (getattr(region, "provenance", {}) or {}).get("role_top_k")
        if role_top_k:
            entry["role_top_k"] = role_top_k
        # OPTIONAL caption text (additive). Present only for a figure/table
        # region whose ``caption_fb_index`` resolved to non-empty text;
        # absent for every other region (byte-stable to baseline). The adapter
        # renders it into ``<figcaption>`` / table ``<caption>`` and uses it as
        # the figure's caption-first accessible name.
        if caption_text is not None:
            entry["caption_text"] = caption_text
        # OPTIONAL structured table grid (additive). The cascade_ir adapter
        # consumes ONLY the distilled provenance list (never the assembler's
        # rich ``<table>`` HTML), so without the grid here a typed ``table``
        # region degraded to a flat ``<p>``. Carry the deterministic cell grid
        # + header/role hints so the adapter can reconstruct a real
        # accessible ``<table>``. Present only for table regions carrying a
        # grid; absent (byte-stable) for every other region.
        if kind == "table":
            grid = payload.get("cell_grid")
            if isinstance(grid, list) and grid:
                entry["cell_grid"] = [list(row or []) for row in grid]
                hdr = payload.get("header_row_indices")
                if isinstance(hdr, (list, tuple)) and hdr:
                    entry["header_row_indices"] = [int(i) for i in hdr]
                cell_roles = payload.get("cell_roles")
                if isinstance(cell_roles, list) and cell_roles:
                    entry["cell_roles"] = [list(r or []) for r in cell_roles]
        # OPTIONAL per-region Stage-5d review key (additive). Present ONLY
        # when the reviewer ran AND corrected/reverted this region; absent
        # when the stage is off (review_verdicts is None) or the region was
        # a pure no-op — keeping the provenance dict byte-stable to baseline.
        # Phase 5 join key: stable min-FB when re-keyed, else positional index
        # (byte-identical legacy path). first_raw already resolves the same
        # min-FB-with-region_index-fallback, so reuse it under the re-key.
        review_key = first_raw if review_regions is not None else region_index
        review = review_index.get(review_key)
        if review is not None:
            entry["review"] = review
        # OPTIONAL pedagogy-class hint (additive). The deterministic
        # structure-correction pass stamps ``payload['css_class']`` on
        # pedagogical headings it demotes to paragraphs (EXAMPLE / Solution /
        # Step / …); surface it here as ``pedagogy_class`` so the Ed4All
        # chunker / retrieval can identify the demoted pedagogical blocks.
        # Present ONLY when the hint is set — byte-stable to baseline when
        # absent (every non-demoted region).
        css_class = payload.get("css_class")
        if isinstance(css_class, str) and css_class.strip():
            entry["pedagogy_class"] = css_class.strip()
        # OPTIONAL OCR-confusable repair (additive; keyed by region_index). The
        # repair pass NEVER mutates ``raw_text`` (the content-hash sourceId basis
        # stays verbatim); it carries the ADDITIVE ``repaired_text`` + ``ocr_repair``
        # edit map so the adapter substitutes the repaired string at render and
        # stamps ``data-dart-repair``. Present ONLY for a region that GAINED >=1
        # gated edit — byte-stable to baseline (both keys absent) when the flag is
        # off or the region was untouched.
        if ocr_repair_edits:
            repair = ocr_repair_edits.get(region_index)
            if isinstance(repair, dict) and repair.get("repaired_text"):
                entry["repaired_text"] = repair["repaired_text"]
                entry["ocr_repair"] = repair["ocr_repair"]
        # OPTIONAL ITEM4 containment key (additive; present only when the
        # SEMANTIK_CONTAINMENT tree was built AND covers this region index).
        # Exposes the derived forest as a flat parent-pointer column
        # (``parent_region_index`` / ``edge_kind``) downstream consumers can
        # read without knowing the sidecar type. Key ABSENT when no tree is
        # threaded (flag off) -> byte-stable to baseline. The adapter ignores
        # unknown keys (the ``review`` block precedent).
        if containment_tree is not None:
            try:
                parents = containment_tree.parent
                kinds = containment_tree.edge_kind
                if 0 <= region_index < len(parents):
                    entry["containment"] = {
                        "parent_region_index": parents[region_index],
                        "edge_kind": kinds[region_index],
                    }
            except (AttributeError, IndexError, TypeError):
                pass
        provenance.append(entry)
    return provenance


# ---------------------------------------------------------------------------
# Phase 6 — bounded, FAIL-SAFE verify-refine loop (SEMANTIK_SECOND_PASS).
#
# Module-level (testable in isolation with a scripted ``run_inner`` + scripted
# endpoint runtime) and lazily importing the reviewer entry points, mirroring
# the cascade's ``clean_structure`` / ``resegment_blocks`` lazy-import posture.
# NEVER ships worse: the Pass-1 snapshot is the structural default return on
# every error / non-convergence / not-better branch (mirroring
# ``lib/retrieval/grounded_answer.py::_apply_completeness_recheck``).
# ---------------------------------------------------------------------------


_TAG_RE = re.compile(r"<[^>]+>")


def _visible_text_tokens(html: str | None) -> list[str]:
    """Lowercased whitespace-token list of the VISIBLE text of an HTML string.

    Strips tags, unescapes the five XML entities the assembler emits, then
    lowercases + whitespace-splits — the document-level token-conservation
    unit for the verify-refine fail-safe (mirrors the reviewer's
    ``_normalize_tokens``). A multiset compare of two such lists answers "did
    the re-assemble drop / duplicate any visible source token?".
    """
    if not html:
        return []
    text = _TAG_RE.sub(" ", str(html))
    for ent, ch in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ):
        text = text.replace(ent, ch)
    return text.lower().split()


def _theta_score_of(report: Any) -> float | None:
    """The composite ``theta_score`` of a ThetaReport, or None when unmeasured.

    ``None`` (stub / not-measured) is treated as "no signal" by the adopt
    gate's not-worse check — theta is a noise-sensitive SECONDARY gate, never
    the primary adopt signal.
    """
    score = getattr(report, "theta_score", None)
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def _doc_gate_passed(lane_fast: dict[str, Any]) -> bool:
    """Stage-10 document-gate pass flag for a ``lane_outputs['fast']`` snapshot."""
    gr = lane_fast.get("gate_result")
    return bool(getattr(gr, "passed", False))


def _round_is_better(
    *,
    reverify: Any,
    restrict_to: frozenset[int],
    prior_lane_fast: dict[str, Any],
    new_lane_fast: dict[str, Any],
    prior_report: Any,
    new_report: Any,
    n_regions: int,
) -> tuple[bool, list[str]]:
    """STRUCTURE-SCOPED better-than predicate for the verify-refine adopt gate.

    Adopt a re-assembled round ONLY if EVERY clause holds (the caller has
    already confirmed ``reverify.passed``):

      1. the flagged regions' failure CLEARED — no region in ``restrict_to`` is
         re-flagged by the re-verification;
      2. the UNFLAGGED regions are BYTE-IDENTICAL to the prior round — the
         primary guard against whole-doc Stage-6 re-author drift (a re-authored
         unflagged region, even text-conserving, FAILS this and forces a
         revert);
      3. the Stage-10 document gate is NOT-WORSE (a prior pass must not become a
         new fail) — noise-tolerant SECONDARY check;
      4. the theta score is NOT-WORSE (None == no signal) — noise-tolerant
         SECONDARY check;
      5. document-level visible text is CONSERVED (multiset equal) — the final
         fail-safe against a dropped / duplicated source token.

    Returns ``(better, reasons)`` where ``reasons`` lists the FAILED clauses
    (empty on adopt) for the audit arm.
    """
    from .assembler.verify_digest import slice_region_html

    reasons: list[str] = []

    # (1) flagged failure cleared.
    reflagged = {
        getattr(fb, "region_index", None) for fb in getattr(reverify, "flagged", ())
    }
    if restrict_to & {i for i in reflagged if isinstance(i, int)}:
        reasons.append("flagged_failure_not_cleared")

    prior_asm = prior_lane_fast.get("assembled")
    new_asm = new_lane_fast.get("assembled")

    # (2) UNFLAGGED regions byte-identical (the load-bearing drift guard).
    for idx in range(n_regions):
        if idx in restrict_to:
            continue
        try:
            if slice_region_html(prior_asm, idx) != slice_region_html(new_asm, idx):
                reasons.append("unflagged_region_drift")
                break
        except Exception:  # noqa: BLE001 — a bad slice is a conservative drift.
            reasons.append("unflagged_region_slice_error")
            break

    # (3) Stage-10 doc gate not-worse.
    if _doc_gate_passed(prior_lane_fast) and not _doc_gate_passed(new_lane_fast):
        reasons.append("doc_gate_worse")

    # (4) theta not-worse (None == no signal; secondary).
    prior_theta = _theta_score_of(prior_report)
    new_theta = _theta_score_of(new_report)
    if prior_theta is not None and new_theta is not None and new_theta < prior_theta:
        reasons.append("theta_worse")

    # (5) document-level visible-text conservation.
    prior_html = getattr(prior_asm, "html", "") if prior_asm is not None else ""
    new_html = getattr(new_asm, "html", "") if new_asm is not None else ""
    if Counter(_visible_text_tokens(prior_html)) != Counter(_visible_text_tokens(new_html)):
        reasons.append("doc_text_not_conserved")

    return (not reasons), reasons


#: The two verifier failure modes the Phase-6b MERGE channel graduates from
#: DETECT-ONLY to FIXABLE (via ``block_resegment.apply_proposed_regroups``).
#: ``FlaggedBlock.fixable`` is False for these (the per-block re-type vocabulary
#: cannot merge), so they ride the SEPARATE merge channel, never the re-drive.
_MERGE_FAILURE_MODES = frozenset({"section_no_body", "example_misordered_from_body"})


def _merge_round_is_better(
    *,
    reverify: Any,
    prior_lane_fast: dict[str, Any],
    new_lane_fast: dict[str, Any],
    prior_report: Any,
    new_report: Any,
) -> tuple[bool, list[str]]:
    """MERGE-channel (Phase 6b) better-than predicate for the adopt gate.

    The re-type channel's per-region BYTE-IDENTITY gate (``_round_is_better``
    clause 2) CANNOT apply across a count-SHRINKING merge — the regions are
    renumbered. So the load-bearing never-regress check here is DOCUMENT-LEVEL
    visible-text conservation (``apply_proposed_regroups`` already
    R-PART/token-conserves at the region level, so a merged region is the
    verbatim concatenation of its members). Adopt iff EVERY clause holds (the
    caller has already confirmed ``reverify.passed``):

      1. no MERGE-mode failure remains — index-shift-SAFE: checked by
         ``failure_mode`` token, NOT by index (the merge renumbers regions, so
         the old flagged indices no longer map);
      2. document-level visible text is CONSERVED (multiset equal) — the PRIMARY
         load-bearing fail-safe for the merge channel;
      3. the Stage-10 doc gate is NOT-WORSE — noise-tolerant SECONDARY;
      4. theta is NOT-WORSE (None == no signal) — noise-tolerant SECONDARY.

    Returns ``(better, reasons)`` (``reasons`` lists the FAILED clauses).
    """
    reasons: list[str] = []

    # (1) merge-mode failures cleared (by mode — robust to the index shift).
    remaining = {
        getattr(fb, "failure_mode", None) for fb in getattr(reverify, "flagged", ())
    }
    if remaining & _MERGE_FAILURE_MODES:
        reasons.append("merge_failure_not_cleared")

    prior_asm = prior_lane_fast.get("assembled")
    new_asm = new_lane_fast.get("assembled")
    prior_html = getattr(prior_asm, "html", "") if prior_asm is not None else ""
    new_html = getattr(new_asm, "html", "") if new_asm is not None else ""

    # (2) document-level visible-text conservation (PRIMARY).
    if Counter(_visible_text_tokens(prior_html)) != Counter(_visible_text_tokens(new_html)):
        reasons.append("doc_text_not_conserved")

    # (3) Stage-10 doc gate not-worse.
    if _doc_gate_passed(prior_lane_fast) and not _doc_gate_passed(new_lane_fast):
        reasons.append("doc_gate_worse")

    # (4) theta not-worse.
    prior_theta = _theta_score_of(prior_report)
    new_theta = _theta_score_of(new_report)
    if prior_theta is not None and new_theta is not None and new_theta < prior_theta:
        reasons.append("theta_worse")

    return (not reasons), reasons


def _verify_refine_loop(
    capped: list[Region],
    review_verdicts: list[Any],
    fast_report: Any,
    lane_outputs: dict[str, dict[str, Any]],
    review_runtime: Any,
    feature_blocks: list[Any],
    state: Any,
    *,
    run_inner: Callable[..., Any],
    log: Callable[[str], None],
) -> tuple[list[Region], list[Any], Any, dict[str, Any]]:
    """Bounded, fail-safe Pass-2 verify-refine loop over the assembled fast lane.

    Returns ``(converged_capped, merged_verdicts, report, signals)``. ``signals``
    is the per-round audit accumulator (the ``second_pass_verify`` arm + the
    Phase-7 capture source); ``signals['adopted']`` tells the cascade whether to
    thread the converged verdicts back with ``review_regions=None`` (a re-typed
    round's verdicts are CAPPED-space).

    FAIL-SAFE: the Pass-1 snapshot (``capped`` + ``review_verdicts`` +
    ``fast_report`` + ``dict(lane_outputs['fast'])``) is the default return on
    every verifier error / non-convergence / not-better branch — the loop NEVER
    ships a doc worse than Pass-1. ``run_inner`` overwrites ``lane_outputs['fast']``
    every call, so the snapshot of ``dict(lane_outputs['fast'])`` is what a revert
    restores.
    """
    from .qwen_specialists.reviewer import (
        resolve_second_pass_rounds,
        run_second_pass_verify,
        run_structure_review,
    )

    # Pass-1 snapshot — the never-ship-worse default (taken BEFORE any re-drive).
    pass1_capped = list(capped)
    pass1_verdicts = list(review_verdicts)
    pass1_report = fast_report
    pass1_lane_fast = dict(lane_outputs["fast"])

    signals: dict[str, Any] = {"adopted": False, "rounds": []}

    def _restore_pass1() -> tuple[list[Region], list[Any], Any, dict[str, Any]]:
        lane_outputs["fast"] = dict(pass1_lane_fast)
        signals["adopted"] = False
        return pass1_capped, pass1_verdicts, pass1_report, signals

    rounds = resolve_second_pass_rounds()

    # ``current_*`` tracks the best-known-good (starts at Pass-1). It is only
    # ever advanced to a round that PASSED re-verification AND was structurally
    # better, so returning ``current_*`` never ships worse.
    current_capped = pass1_capped
    current_verdicts = pass1_verdicts
    current_report = pass1_report
    current_lane_fast = pass1_lane_fast

    for round_idx in range(1, rounds + 1):
        row: dict[str, Any] = {"round": round_idx}
        signals["rounds"].append(row)

        # (a) verify the CURRENT assembled doc.
        try:
            verdict = run_second_pass_verify(
                current_lane_fast.get("assembled"),
                current_capped,
                feature_blocks,
                review_runtime,
                council_state=state,
            )
        except Exception as exc:  # noqa: BLE001 — fail-safe: keep Pass-1.
            log(f"[cascade] verify-refine r{round_idx}: verifier error ({exc}) -> keep Pass-1")
            row.update(error=str(exc), passed=None, adopted=False)
            return _restore_pass1()

        flagged = list(getattr(verdict, "flagged", ()) or ())
        fixable = [fb for fb in flagged if getattr(fb, "fixable", False)]
        deferred = [fb for fb in flagged if not getattr(fb, "fixable", False)]
        row.update(
            passed=bool(verdict.passed),
            flagged=len(flagged),
            fixable_indices=sorted(getattr(fb, "region_index", -1) for fb in fixable),
            deferred_modes=sorted({getattr(fb, "failure_mode", "?") for fb in deferred}),
        )

        # (b) clean pass -> converged on ``current`` (Pass-1 or an adopted round).
        if verdict.passed:
            row["adopted"] = current_capped is not pass1_capped
            lane_outputs["fast"] = dict(current_lane_fast)
            signals["adopted"] = current_capped is not pass1_capped
            return current_capped, current_verdicts, current_report, signals

        # (c) No re-type-fixable flags remain. Either the Phase-6b MERGE
        # channel handles the deferred merge modes, or there is nothing this
        # index-keyed reviewer can do -> keep ``current`` (= Pass-1 unless an
        # earlier round adopted).
        if not fixable:
            # --- Phase 6b MERGE channel ----------------------------------
            # Runs as its OWN round (NEVER mixed with a re-type re-drive in one
            # re-assemble): the round's flagged set carries ONLY deferred merge
            # modes here (re-type rounds fire FIRST, while fixable flags exist).
            # Gated on SEMANTIK_UNIT_REGROUP; the merge SHRINKS len(capped), so
            # the re-type channel's len-invariant check above is scoped to the
            # re-type branch and intentionally does NOT apply on this path.
            from .qwen_specialists.block_resegment import (
                apply_proposed_unit_fix,
                resolve_unit_regroup_mode,
            )

            merge_runs = [
                tuple(getattr(fb, "proposed_regroup_run", ()) or ())
                for fb in deferred
                if (getattr(fb, "proposed_regroup_run", ()) or ())
                and getattr(fb, "failure_mode", None) in _MERGE_FAILURE_MODES
            ]

            if not (resolve_unit_regroup_mode() and merge_runs):
                # No merge channel available (regroup off, or no proposed runs)
                # -> the deferred modes are un-fixable here.
                row.update(adopted=False, deferred_only=True)
                lane_outputs["fast"] = dict(current_lane_fast)
                signals["adopted"] = current_capped is not pass1_capped
                return current_capped, current_verdicts, current_report, signals

            row["merge_runs"] = [list(r) for r in merge_runs]

            # (c1) Apply the proposed unit fix through the SAME R-PART/token
            # conservation gates the deterministic detector rides. NEVER raises;
            # a hallucinated / over-broad run is dropped per-op. ITEM3 Phase 3:
            # a NON-contiguous run (previously silently dropped) is decomposed
            # into MoveOps + a contiguous merge when SEMANTIK_MOVE_OP=live;
            # non-live delegates to apply_proposed_regroups (byte-identical drop).
            merged_capped, ops = apply_proposed_unit_fix(
                current_capped, feature_blocks, merge_runs
            )
            if not ops or len(merged_capped) == len(current_capped):
                # No op landed (every proposed run dropped by the gates) -> the
                # merge didn't help. Keep ``current`` + STOP (re-applying a
                # deterministic no-op is pointless) -> never ship worse.
                log(
                    f"[cascade] verify-refine r{round_idx}: merge produced no "
                    f"landed op -> keep current"
                )
                row.update(adopted=False, merge_noop=True)
                lane_outputs["fast"] = dict(current_lane_fast)
                signals["adopted"] = current_capped is not pass1_capped
                return current_capped, current_verdicts, current_report, signals

            row["merge_ops"] = len(ops)

            # (c2) Re-assemble the MERGED (shorter) list (overwrites fast).
            merged_report = run_inner("fast", regions=merged_capped)
            merged_lane_fast = dict(lane_outputs["fast"])

            # (c3) Re-verify the new assembled doc.
            try:
                merge_reverify = run_second_pass_verify(
                    merged_lane_fast.get("assembled"),
                    merged_capped,
                    feature_blocks,
                    review_runtime,
                    council_state=state,
                )
            except Exception as exc:  # noqa: BLE001 — fail-safe: keep current.
                log(
                    f"[cascade] verify-refine r{round_idx}: merge re-verify error "
                    f"({exc}) -> keep current"
                )
                row.update(error=str(exc), adopted=False, merge_adopted=False)
                lane_outputs["fast"] = dict(current_lane_fast)
                signals["adopted"] = current_capped is not pass1_capped
                return current_capped, current_verdicts, current_report, signals

            if not merge_reverify.passed:
                # Re-verify STILL failing -> the merge didn't fully resolve.
                # Discard it, restore ``current`` for the next round, retry
                # (bounded). On exhaustion the loop returns ``current``.
                log(
                    f"[cascade] verify-refine r{round_idx}: merge re-verify still "
                    f"failing -> retry"
                )
                row.update(adopted=False, merge_adopted=False, retry=True)
                lane_outputs["fast"] = dict(current_lane_fast)
                continue

            merge_better, merge_reasons = _merge_round_is_better(
                reverify=merge_reverify,
                prior_lane_fast=current_lane_fast,
                new_lane_fast=merged_lane_fast,
                prior_report=current_report,
                new_report=merged_report,
            )
            row.update(merge_reverify_passed=True, merge_better=merge_better, merge_reasons=merge_reasons)

            if merge_better:
                # ADOPT the merge. The count SHRANK, so the prior verdict list no
                # longer aligns 1:1 with the merged region list. SAFEST
                # provenance handling (cannot corrupt): thread the verdicts back
                # as None so ``_build_region_provenance`` attaches NO (stale)
                # review sub-block to a merged region -> one entry per merged
                # region keyed by its region_index, no IndexError, no
                # misattribution. ``merge_adopted`` records the distinct action.
                current_capped = merged_capped
                current_verdicts = None
                current_report = merged_report
                current_lane_fast = merged_lane_fast
                lane_outputs["fast"] = dict(merged_lane_fast)
                row.update(adopted=True, merge_adopted=True)
                signals["adopted"] = True
                signals["merge_adopted"] = True
                continue

            # Re-verify PASSED but the merge is not-better (doc-text / gate /
            # theta worse) -> a deterministic re-apply won't improve it. Keep
            # ``current`` (best-known-good) + STOP -> never ship the merge.
            log(
                f"[cascade] verify-refine r{round_idx}: merge round not-better "
                f"({merge_reasons}) -> keep current"
            )
            row.update(adopted=False, merge_adopted=False)
            lane_outputs["fast"] = dict(current_lane_fast)
            signals["adopted"] = current_capped is not pass1_capped
            return current_capped, current_verdicts, current_report, signals

        restrict_to = frozenset(
            int(getattr(fb, "region_index")) for fb in fixable
        )
        feedback = {
            int(getattr(fb, "region_index")): getattr(fb, "fix_hint", "") or ""
            for fb in fixable
        }
        row["restrict_to"] = sorted(restrict_to)

        # (d) targeted re-review over the CURRENT capped list (CAPPED space).
        try:
            new_capped, new_verdicts = run_structure_review(
                current_capped,
                feature_blocks,
                review_runtime,
                council_state=state,
                restrict_to=restrict_to,
                feedback_by_idx=feedback,
            )
        except Exception as exc:  # noqa: BLE001 — fail-safe: keep Pass-1.
            log(f"[cascade] verify-refine r{round_idx}: re-review error ({exc}) -> keep Pass-1")
            row.update(error=str(exc), adopted=False)
            return _restore_pass1()

        # Re-type channel invariant: the region COUNT never changes (Phase-6b
        # owns the count-shrinking merge channel). A drifted count -> keep
        # ``current`` rather than risk a corrupt re-assemble.
        if len(new_capped) != len(current_capped):
            log(
                f"[cascade] verify-refine r{round_idx}: region count changed "
                f"({len(current_capped)} -> {len(new_capped)}) -> keep current"
            )
            row.update(adopted=False, len_changed=True)
            lane_outputs["fast"] = dict(current_lane_fast)
            signals["adopted"] = current_capped is not pass1_capped
            return current_capped, current_verdicts, current_report, signals

        # (e) re-assemble the re-reviewed list (overwrites lane_outputs['fast']).
        new_report = run_inner("fast", regions=new_capped)
        new_lane_fast = dict(lane_outputs["fast"])

        # (f) re-verify the NEW assembled doc.
        try:
            reverify = run_second_pass_verify(
                new_lane_fast.get("assembled"),
                new_capped,
                feature_blocks,
                review_runtime,
                council_state=state,
            )
        except Exception as exc:  # noqa: BLE001 — fail-safe: keep Pass-1.
            log(f"[cascade] verify-refine r{round_idx}: re-verify error ({exc}) -> keep Pass-1")
            row.update(error=str(exc), adopted=False)
            return _restore_pass1()

        better, reasons = _round_is_better(
            reverify=reverify,
            restrict_to=restrict_to,
            prior_lane_fast=current_lane_fast,
            new_lane_fast=new_lane_fast,
            prior_report=current_report,
            new_report=new_report,
            n_regions=len(new_capped),
        )
        row.update(reverify_passed=bool(reverify.passed), better=better, reasons=reasons)

        if reverify.passed and better:
            # ADOPT — advance ``current`` to the re-typed round; the next round's
            # top-of-loop verify confirms it (passes) and returns it.
            current_capped = new_capped
            current_verdicts = new_verdicts
            current_report = new_report
            current_lane_fast = new_lane_fast
            lane_outputs["fast"] = dict(new_lane_fast)
            row["adopted"] = True
            signals["adopted"] = True
            continue

        if reverify.passed and not better:
            # Re-verify passed but the round DRIFTED an unflagged region / theta /
            # text -> dangerous whole-doc re-author noise. REVERT to Pass-1 and
            # stop (retrying won't help; never ship the drift).
            log(
                f"[cascade] verify-refine r{round_idx}: re-typed round not-better "
                f"({reasons}) -> revert to Pass-1"
            )
            row["adopted"] = False
            return _restore_pass1()

        # (g) re-verify STILL failed -> the targeted fix didn't fully land.
        # Discard the round, restore ``current`` for the next round's verify, and
        # retry (bounded). On exhaustion the loop returns ``current`` (= Pass-1).
        log(f"[cascade] verify-refine r{round_idx}: re-verify still failing -> retry")
        row.update(adopted=False, retry=True)
        lane_outputs["fast"] = dict(current_lane_fast)

    # Exhausted the round bound without a clean converge. Keep the best-known-
    # good (= Pass-1 unless an adopted round was already returned above).
    lane_outputs["fast"] = dict(current_lane_fast)
    signals["adopted"] = current_capped is not pass1_capped
    return current_capped, current_verdicts, current_report, signals


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
        pdf_path=pdf_path,
    )
    stages["stage1_5"] = time.perf_counter() - t

    # ------------------------------------------------------------------
    # Stage 5d — 70B text-preserving STRUCTURE REVIEWER (default OFF).
    #
    # Runs AFTER build_structure_graph and STRICTLY BEFORE Stage 5b/5c
    # enrichment + the per-kind cap, so a 5d re-tag INTO table/figure is
    # picked up by 5b/5c, and corrected kinds re-route Stage-6 + assembly.
    # Gated by SEMANTIK_STRUCTURE_REVIEW; OFF -> pure pass-through (the
    # variables below stay None/empty and every downstream join is a no-op,
    # byte-identical to a no-Stage-5d run).
    # ------------------------------------------------------------------
    review_verdicts: list[Any] | None = None
    stage5d_metadata_drop_ids: frozenset[int] = frozenset()

    # Stage-5e block JOIN/SPLIT audit — list of applied-op dicts when the pass
    # ran (default OFF -> stays None, so the audit section is byte-stable).
    resegment_ops_audit: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Stage 5d-det — DETERMINISTIC structure correction (default ON).
    #
    # Runs BEFORE the 70B Stage-5d reviewer (and before the cap + assembly),
    # so BOTH assembled.html and region_provenance pick up the corrected
    # regions (they derive from this same list). Fixes the front-matter /
    # phantom-TOC / OCR-noise drops + the pedagogical-label heading
    # over-nesting the 70B reviewer's prompt explicitly defers to "a separate
    # DETERMINISTIC pass". Gated by SEMANTIK_STRUCTURE_CLEAN; OFF -> byte-
    # identical pass-through. Reuses the reviewer's invariant harness
    # (FB-partition-immutable re-tag + token-conservation + fail-closed).
    # ------------------------------------------------------------------
    from .qwen_specialists.deterministic_structure import (
        clean_structure,
        resolve_promote_section_headings_mode,
        resolve_structure_clean_mode,
    )

    if resolve_structure_clean_mode():
        t = time.perf_counter()
        structure_regions, structure_clean_diag = clean_structure(
            structure_regions,
            feature_blocks,
        )
        stages["stage5d_det"] = time.perf_counter() - t
        log(
            "[cascade] Stage 5d-det (deterministic structure correction): "
            f"front_matter_dropped={structure_clean_diag.get('front_matter_dropped')}, "
            f"pedagogical_demoted={structure_clean_diag.get('pedagogical_demoted')}, "
            f"headings {structure_clean_diag.get('headings_before')} -> "
            f"{structure_clean_diag.get('headings_after')}"
            + (
                " (REVERTED: token-conservation)"
                if structure_clean_diag.get("reverted_for_invariant")
                else ""
            )
        )

    # Snapshot the (possibly det-corrected) structure regions for the C2
    # FB-survival baseline (the flag-OFF region list the cap is compared
    # against). Captured AFTER the deterministic pass so the 70B reviewer's
    # baseline reflects the corrected region list.
    pre_review_regions = list(structure_regions)

    # Phased GPU (one model resident at a time — the 8GB-card invariant): evict
    # the council BERTs NOW. Their signals are already baked into ``state`` + the
    # (det-corrected) structure regions, and NOTHING downstream needs the
    # backbone WEIGHTS on the GPU — the Stage-5d reviewer + Stage-6 specialists
    # ride the ollama/endpoint seat in a SEPARATE process, theta is its own
    # DeBERTa, the rerankers are heuristic. Without this early eviction the
    # in-process council backbones (~2-3 GB) coexist with the endpoint reviewer's
    # resident 7B during Stage-5d (the first ollama call) and contend/OOM the
    # card. The pre-Stage-6 eviction below then becomes an idempotent no-op
    # safety net. (Proven safe by the standalone reviewer/render harnesses, which
    # release here before any ollama call.)
    from .council.base import release_council_gpu as _release_council_pre_review

    _release_council_pre_review()

    if resolve_structure_review_mode():
        from .qwen_specialists.reviewer import run_structure_review
        from .qwen_specialists.runtime import (
            make_runtime,
            resolve_structure_review_model,
        )

        log(
            f"[cascade] running Stage 5d (70B structure review, "
            f"{len(structure_regions)} regions)"
        )
        t = time.perf_counter()
        # The reviewer rides the SAME hosted-70B endpoint seat as the
        # Stage-6 endpoint specialists; it only needs generate_batch, which
        # make_runtime('endpoint') supplies. NO local GGUF / GPU here. The
        # reviewer pins its OWN model via resolve_structure_review_model()
        # (SEMANTIK_STRUCTURE_REVIEW_MODEL > SEMANTIK_SPECIALIST_MODEL >
        # NVIDIA_LARGE_MODEL > default) so the documented reviewer-specific
        # seat actually routes the call — unset reviewer model falls back to
        # the shared specialist model, identical to the legacy behaviour.
        review_runtime = make_runtime(
            "endpoint", model=resolve_structure_review_model()
        )
        reviewed_regions, review_verdicts = run_structure_review(
            structure_regions,
            feature_blocks,
            review_runtime,
            council_state=state,
        )
        # Which regions did THIS stage NEWLY re-tag to metadata_drop?
        # (a region that was already metadata_drop pre-review is not a
        # Stage-5d drop and is NOT cap-exempt). These are exempt from the
        # per-kind cap's content buckets (C2). Keyed by the region's STABLE
        # min-FB (Phase 5) — NOT its positional index — so the exemption
        # survives the Stage-5e N->1 regroup region-count reduction (a merged
        # region inherits the anchor's min-FB; a metadata_drop region is never
        # a regroup anchor/body, so its min-FB key is stable).
        stage5d_metadata_drop_ids = frozenset(
            min_fb
            for before, after in zip(pre_review_regions, reviewed_regions)
            if after.kind == "metadata_drop"
            and before.kind != "metadata_drop"
            and (min_fb := _region_min_fb(after)) is not None
        )
        structure_regions = reviewed_regions
        stages["stage5d"] = time.perf_counter() - t
        n_corrected = sum(
            1
            for v in (review_verdicts or ())
            if getattr(v, "kind_before", None) != getattr(v, "kind_after", None)
            or getattr(v, "level_before", None) != getattr(v, "level_after", None)
        )
        log(f"[cascade] Stage 5d corrected {n_corrected} region(s)")

    # ------------------------------------------------------------------
    # Stage 5e — block JOIN / SPLIT (FB-boundary re-partition; default OFF).
    #
    # Runs AFTER both the deterministic (5d-det) + 70B (5d) re-tag passes and
    # STRICTLY BEFORE Stage 5b/5c enrichment + the per-kind cap, so a merged /
    # split region is picked up by 5b/5c and re-routes Stage-6 + assembly.
    # Gated by SEMANTIK_BLOCK_RESEGMENT; OFF -> pure pass-through (byte-
    # identical to a no-Stage-5e run). Deterministic-first: the council
    # MergeOrSplit head + the pedagogical-label regex + the cross-page
    # continuation cue are load-bearing; the optional 70B layer
    # (SEMANTIK_BLOCK_RESEGMENT_LLM) proposes EXTRA ops re-validated by the
    # same R-PART + token-conservation gates. Fails closed to the input list.
    # ------------------------------------------------------------------
    from .qwen_specialists.block_resegment import (
        build_resegment_audit_rows,
        resegment_blocks,
        resolve_block_resegment_llm_mode,
        resolve_block_resegment_mode,
        resolve_move_op_mode,
        resolve_split_fused_section_titles_mode,
        resolve_unit_regroup_mode,
    )

    # Widened gate (Phase 6 + lane B + ITEM3): enter Stage-5e when the same-kind
    # block resegment OR the cross-kind pedagogical-unit regroup OR the
    # fused-section-title split OR the MOVE arm is on. Because per-flag gating
    # lives INSIDE resegment_blocks, this gate only DECIDES WHETHER to enter
    # Stage-5e — it does not leak one flag's arm into another flag's run.
    # ITEM3 note: ``resolve_move_op_mode() != "off"`` defaults to shadow, and
    # post-ITEM1 (regroup default-ON) the regroup clause already enters Stage-5e
    # by default, so this widening is a practical NO-OP for a default run; it
    # only matters when the regroup is reverted (SEMANTIK_UNIT_REGROUP=0) but the
    # MOVE arm is still on (audit-only shadow rows — document output bytes
    # unchanged, since shadow never mutates the region list). All-flags-off
    # (including SEMANTIK_MOVE_OP=0) stays byte-identical to the legacy gate.
    if (
        resolve_block_resegment_mode()
        or resolve_unit_regroup_mode()
        or resolve_split_fused_section_titles_mode()
        or resolve_move_op_mode() != "off"
    ):
        t = time.perf_counter()
        # The LLM layer rides the SAME hosted-70B endpoint seat as Stage-5d /
        # Stage-6; it only needs generate_batch. NO local GGUF / GPU here. The
        # runtime is built ONLY when the LLM layer is on (deterministic-only
        # runs never touch the endpoint).
        resegment_runtime = None
        if resolve_block_resegment_llm_mode():
            from .qwen_specialists.runtime import make_runtime

            resegment_runtime = make_runtime("endpoint")
        log(
            f"[cascade] running Stage 5e (block join/split, "
            f"{len(structure_regions)} regions, "
            f"llm={'on' if resegment_runtime is not None else 'off'})"
        )
        resegmented_regions, resegment_ops = resegment_blocks(
            structure_regions,
            feature_blocks,
            state,
            runtime=resegment_runtime,
        )
        structure_regions = resegmented_regions
        # ITEM6 — re-stamp role distributions: Stage-5e merge/split/regroup mints
        # fresh Regions whose provenance lacks role_top_k. Idempotent for
        # untouched regions (a merged region's min(feature_block_indices) is the
        # label region's rep-FB).
        structure_regions = stamp_role_distributions(structure_regions, state)
        # Audit section — the op list (op type, source ids, conservation flag).
        # Parallel to the structure_review audit (~L948). Stays None when off.
        # Phase 9: a cross-kind pedagogical-unit op (subtype='regroup') records
        # op='regroup' + the merged unit's semantic_class + regions_folded so
        # the block_resegment DecisionCapture can interpolate regroup
        # counts/classes into a dynamic, replayable rationale — with NO
        # decision_event schema change (block_resegment is already in the enum).
        # A same-kind op (subtype default 'merge') keeps op=op.op and the
        # original 4-key row VERBATIM (byte-stable when the regroup flag is off —
        # no regroup op exists so no row gains the extra keys). Built via the
        # single SoT helper so the cascade and its tests never drift.
        resegment_ops_audit = build_resegment_audit_rows(resegment_ops)
        # Decision-capture EMIT is WIRED at the Ed4All boundary (owned by the
        # parent, not here): MCP/tools/pipeline_tools.py section "2c" calls
        # _emit_block_resegment_capture(_semantik_resolve_block_resegment(result))
        # after every conversion — the resolver reads this audit off the
        # result-dict top-level key ("block_resegment") for the in-process arm
        # and off the bridge JSON for the cross-venv arm (run_cascade_json.py
        # _build_bridge_dict forwards it), so ONE block_resegment DecisionCapture
        # fires per converted document, mirroring the structure_review emit.
        stages["stage5e"] = time.perf_counter() - t
        n_merges = sum(1 for o in resegment_ops if o.op == "merge")
        n_splits = sum(1 for o in resegment_ops if o.op == "split")
        log(
            f"[cascade] Stage 5e applied {n_merges} merge(s), "
            f"{n_splits} split(s)"
        )

    # ------------------------------------------------------------------
    # Post-Stage-5e — fused-title child PROMOTION (lane B; default OFF x2).
    #
    # The clean_structure sub-pass E promotion runs at Stage-5d-det (BEFORE
    # Stage-5e), so it can never see a title child the Stage-5e fused-title
    # SPLIT only just carved off. This narrow hook promotes that split-off
    # title child (child_index==0 of a subtype='fused_title' split) to a
    # section heading. Gated on BOTH SEMANTIK_SPLIT_FUSED_SECTION_TITLES (the
    # split that created the child) AND SEMANTIK_PROMOTE_SECTION_HEADINGS (the
    # promotion opt-in): split-on/promote-off leaves the title child a
    # paragraph (still an improvement; promotable later by rerender);
    # promote-on/split-off is byte-identical to today (no fused-title child
    # exists, so this is a natural no-op). Fail-closed whole-revert on
    # token-conservation (mirrors clean_structure).
    # ------------------------------------------------------------------
    if (
        resolve_split_fused_section_titles_mode()
        and resolve_promote_section_headings_mode()
    ):
        from .qwen_specialists.deterministic_structure import (
            promote_fused_title_children,
        )

        structure_regions, fused_promote_diag = promote_fused_title_children(
            structure_regions, feature_blocks
        )
        log(
            "[cascade] post-5e fused-title promotion: "
            f"promoted={fused_promote_diag.get('promoted')}"
            + (
                " (REVERTED: token-conservation)"
                if fused_promote_diag.get("reverted_for_invariant")
                else ""
            )
        )

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
    detect_figures = resolve_detect_figures()
    n_figs_before = sum(1 for r in structure_regions if r.kind == "figure")
    if n_figs_before:
        log(f"[cascade] running Stage 5c (figure-bbox PNG rendering, n_figures={n_figs_before})")
        t = time.perf_counter()
        from .image_extract import render_figure_regions_to_bytes

        # Part F — fail-soft per-region when the figure path is on (one
        # bad bbox on a 39-image page must not abort the doc); the legacy
        # demote path stays loud (fail_soft=False).
        structure_regions = render_figure_regions_to_bytes(
            structure_regions,
            feature_blocks,
            pdf_path,
            fail_soft=detect_figures,
        )
        stages["stage5c"] = time.perf_counter() - t

    capped = _cap_regions_per_kind(
        structure_regions,
        cap=max_regions_per_kind,
        stage5d_metadata_drop_ids=stage5d_metadata_drop_ids,
    )

    # C2 / R12 — cap-safety FB-survival assertion. When Stage 5d ran, prove
    # no real-content FeatureBlock that survived the cap on the flag-OFF
    # baseline was EVICTED by a re-role pushing it past the cap. Fail closed
    # (revert the Stage-5d corrections for this run) rather than ship a
    # document with lost content. Skipped cleanly when the reviewer is off
    # (review_verdicts is None) or the cap is disabled (cap <= 0).
    if review_verdicts is not None and max_regions_per_kind > 0:
        baseline_capped = _cap_regions_per_kind(
            pre_review_regions,
            cap=max_regions_per_kind,
        )
        try:
            _assert_fb_survival_through_cap(baseline_capped, capped)
        except CapSafetyError as exc:
            log(f"[cascade] Stage 5d cap-safety FAILED, reverting corrections: {exc}")
            # Fail closed: discard the Stage-5d corrections + verdicts for
            # this run and re-cap the un-reviewed (flag-OFF) region list.
            structure_regions = pre_review_regions
            review_verdicts = None
            stage5d_metadata_drop_ids = frozenset()
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
    # Part F — caption deferral. When the figure path is on and captioning
    # is deferred (auto with no CUDA, or explicit off), SKIP Stage 6b so
    # the figure ships the honest type-level ``"Figure."`` alt (via the
    # assembler's guard_figure_alt) instead of loading SmolVLM2. The legacy
    # demote-figure path (figure flag off) always captions, as before.
    caption_figures = (not detect_figures) or _figure_captioning_active()
    if n_figs_capped and caption_figures:
        log(f"[cascade] running Stage 6b (figure captioner, n_figures={n_figs_capped})")
        t = time.perf_counter()
        from .figure_captioner import caption_figure_regions

        capped = caption_figure_regions(capped)
        stages["stage6b"] = time.perf_counter() - t
        # TORCH LEASE — reclaim the SmolVLM2 captioner's allocator cache after
        # captioning completes (its weights drop by scope; this hands the card
        # cleanly to the Stage-6 GGUF). Gated on ED4ALL_GPU_LIFECYCLE, fail-soft.
        _gpu_lifecycle_release(torch=True, stage="post-Stage-6b-captioner")
    elif n_figs_capped:
        log(
            f"[cascade] Stage 6b DEFERRED (figure captioning off; "
            f"n_figures={n_figs_capped} ship type-level alt)"
        )

    # ------------------------------------------------------------------
    # Stage F — figure sidecar write + ``image_src`` wiring (Part F).
    # After captioning, write each figure region's ``image_png_bytes`` to
    # a deterministic sidecar PNG (``{stem}_figures/fig-{first_fb}.png``)
    # and stamp ``payload["image_src"]`` = ``./{stem}_figures/fig-N.png``
    # so the emitters fill the previously-empty ``<img src="">``. Only the
    # ``image_src`` + ``figure_alt`` travel onward; the raw PNG bytes are
    # written to disk and STRIPPED before the JSON bridge. Per-region
    # fail-soft (a write failure degrades that one figure to no src).
    # No-op when the figure path is off (byte-stable).
    # ------------------------------------------------------------------
    if detect_figures and n_figs_capped:
        t = time.perf_counter()
        capped = _write_figure_sidecars(capped, pdf_path, log=log)
        stages["stage_f_sidecar"] = time.perf_counter() - t

    # Evict the council BERTs from the GPU before Stage 6 loads the Qwen GGUF.
    # Stages 6-12 don't use the shared backbone (Qwen is llama-cpp, theta is its
    # own DeBERTa, the soft rerankers are heuristic), so this is safe and frees
    # ~2-3 GB — without it, large PDFs OOM on GGUF load on the 8 GB card even in
    # an isolated process ("one model resident at a time" across BERT->Qwen).
    from .council.base import release_council_gpu

    release_council_gpu()

    # OLLAMA LEASE #1 — hand the card off from the Stage-5d reviewer +
    # Stage-5e resegment-LLM ollama seat BEFORE the Stage-6 GGUF loads. The
    # 5d+5e run is ONE ollama-consumer window (same seat/model): releasing
    # between 5d and 5e would be exactly the cold-reload churn the lease
    # directive forbids, so the release fires here, at the END of that window
    # (the missing piece the lane names "after the Stage-5d reviewer
    # completes"). No-op unless SEMANTIK_STRUCTURE_REVIEW / _BLOCK_RESEGMENT
    # actually loaded a model (otherwise the /api/ps sweep finds nothing);
    # gated on ED4ALL_GPU_LIFECYCLE, fail-soft, lazy-reload-safe.
    _gpu_lifecycle_release(ollama=True, stage="post-Stage-5e/pre-Stage-6")

    # GRACEFUL-STOP SEAM (a) — post-Stage-5e / pre-Stage-6, the cheapest-loss
    # cooperative checkpoint point: everything upstream (extract / council /
    # structure / 5b-5e) is disk-cache-recoverable, and Stage 6 is the
    # ~3h/chapter local-7B authoring exposure. When the Ed4All side handed in a
    # SEMANTIK_STOP_SENTINEL and it now exists, raise CascadeStopRequested here
    # so we never enter Stage 6. No-op (byte-identical) when no sentinel path
    # was provided; the probe is fail-soft. Mirrors the _gpu_lifecycle_release
    # cross-venv twin — the seam polls a PATH handed in from outside, never
    # importing Ed4All's lib/.
    check_cascade_stop("cascade:post-stage5e-pre-stage6")

    # ------------------------------------------------------------------
    # Stages 6-12 are encapsulated so the offline-retry orchestrator can
    # re-run the inner pipeline against ``lane="offline"``. Stage outputs
    # that the LAST lane produced are captured into ``lane_outputs`` so
    # the per-PDF JSON report describes the lane that ultimately won
    # (matched by the FINAL ThetaReport's ``lane`` field).
    # ------------------------------------------------------------------
    lane_outputs: dict[str, dict[str, Any]] = {}

    def _run_inner(lane: str, regions: list[Region] | None = None) -> Any:
        """Stages 6-12 for a given lane. Returns a fresh ThetaReport.

        Side-effect: populates ``lane_outputs[lane]`` with the artifacts
        downstream code needs (axe summary, assembled doc, etc.).
        Per-stage timings are recorded with a ``_offline`` suffix on the
        offline pass so totals don't collide.

        Phase 6 (verify-refine): ``regions`` defaults to the enclosing
        post-cap ``capped``. ``regions is None`` is byte-identical to every
        legacy call site (fast / offline lanes both pass no regions); the
        verify-refine loop passes a RE-REVIEWED region list explicitly so the
        returned ``lane_outputs[lane]['assembled']`` is provably the document
        it just re-typed (the closure-rebind seam). EVERY ``capped`` read in
        this body binds to ``regions`` — a missed read would silently
        re-assemble the unchanged pass-1 regions against a re-typed list.
        """
        suffix = "" if lane == "fast" else "_offline"
        # Phase 6 closure-rebind: default to the enclosing post-cap ``capped``
        # (byte-identical legacy behaviour) or the explicitly-passed re-reviewed
        # list. Bound ONCE here so every read below is the same list.
        regions = capped if regions is None else regions

        log(
            f"[cascade] [{lane}] running Stage 6 ({runtime_mode} runtime, k={k}) "
            f"on {len(regions)} regions"
        )
        t_inner = time.perf_counter()
        cands = run_qwen_specialists(
            regions,
            feature_blocks,
            k=k,
            runtime_mode=runtime_mode,
            lane=lane,  # type: ignore[arg-type]
        )
        stages[f"stage6{suffix}"] = time.perf_counter() - t_inner
        # TORCH LEASE (belt-and-suspenders) — the Stage-6 llama-cpp AdapterSwap
        # already frees each GGUF adapter group on __exit__ (runtime.free →
        # del+gc+empty_cache, including the last group); this reclaims any
        # residual allocator cache before the deterministic Stages 7-11. Gated +
        # fail-soft; lazy-reload-safe on the offline-retry re-entry of this
        # closure.
        _gpu_lifecycle_release(torch=True, stage=f"post-Stage-6[{lane}]")

        n_in = sum(len(v) for v in cands.values())

        # Stage 6c — deterministic H43 enrichment (Plan 12 B1): complex
        # tables (multi header rows / dual axis / spanned <th>) get
        # id/headers association generated from grid arithmetic BEFORE
        # the gate, which now requires it on complex tables. Simple
        # tables pass through byte-identical.
        cands = _apply_h43_to_table_candidates(cands, regions)

        log(f"[cascade] [{lane}] running Stage 7 (per-region hard gate)")
        t_inner = time.perf_counter()
        survs, res = gate_per_region(
            cands,
            regions,
            feature_blocks,
            validator=validator,
        )
        stages[f"stage7{suffix}"] = time.perf_counter() - t_inner
        n_out = sum(len(v) for v in survs.values())
        s7_dist = _stage7_violation_distribution(res)
        s7_skipped = _stage7_skip_distribution(res)

        log(f"[cascade] [{lane}] running Stage 8 (per-region soft reranker)")
        t_inner = time.perf_counter()
        top = rerank_per_region(survs, regions, feature_blocks)
        stages[f"stage8{suffix}"] = time.perf_counter() - t_inner

        log(f"[cascade] [{lane}] running Stage 9 (assembler, {runtime_mode} runtime)")
        t_inner = time.perf_counter()
        asm = assemble_document(
            top,
            regions,
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
        ro_at_risk = _reading_order_at_risk_indices(top, list(regions))
        rep = evaluate(
            asm,
            feature_blocks=feature_blocks,
            regions=regions,
            wcag_status=wcag,
            lane=lane,  # type: ignore[arg-type]
            stage11_scored=sc,
            reading_order_at_risk=ro_at_risk,
        )
        stages[f"stage12{suffix}"] = time.perf_counter() - t_inner
        # TORCH LEASE — reclaim the theta DeBERTa cross-encoder's allocator
        # cache after Stage-12 evaluate. Theta loads per-evaluate and drops its
        # refs by scope; this hands the card over at the seam. Idempotent +
        # lazy-reload-safe: maybe_offline_retry re-enters this closure and theta
        # reloads on the next evaluate. Gated on ED4ALL_GPU_LIFECYCLE, fail-soft.
        _gpu_lifecycle_release(torch=True, stage=f"post-Stage-12-theta[{lane}]")

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

    # ------------------------------------------------------------------
    # Stage 9-verify (Phase 6) — bounded, FAIL-SAFE verify-refine loop.
    #
    # The single point where the post-cap region list (``capped``) and the
    # assembled fast-lane document (``lane_outputs['fast']['assembled']``)
    # coexist. Gated by SEMANTIK_SECOND_PASS AND a Pass-1 reviewer having run
    # (``review_verdicts is not None`` — no Pass-1 verdicts means nothing to
    # bounce back to). OFF -> byte-identical: no verifier prompt, no extra
    # assemble round, no ``second_pass_verify`` audit arm.
    #
    # The loop verifies the assembled doc, bounces ONLY the FIXABLE flagged
    # region indices back through the EXISTING Pass-1 reviewer (re-type keyed by
    # index, never text), re-assembles via ``_run_inner('fast', regions=...)``,
    # and ADOPTS a round only when it PASSES re-verification AND is structurally
    # better (flagged failure cleared, UNFLAGGED regions byte-identical, doc gate
    # / theta not-worse, visible text conserved). On any error / non-convergence
    # / not-better it keeps the Pass-1 snapshot (never ships worse). ``capped``
    # and ``review_verdicts`` are rebound to the converged values; the offline
    # retry below reads the converged ``capped`` (INHERITS it).
    # ------------------------------------------------------------------
    second_pass_signals: dict[str, Any] | None = None
    second_pass_adopted = False
    from .qwen_specialists.reviewer import resolve_second_pass_mode

    if resolve_second_pass_mode() and review_verdicts is not None:
        log("[cascade] running Stage 9-verify (Pass-2 verify-refine loop)")
        t = time.perf_counter()
        (
            capped,
            review_verdicts,
            fast_report,
            second_pass_signals,
        ) = _verify_refine_loop(
            capped,
            review_verdicts,
            fast_report,
            lane_outputs,
            review_runtime,
            feature_blocks,
            state,
            run_inner=_run_inner,
            log=log,
        )
        stages["stage9_verify"] = time.perf_counter() - t
        second_pass_adopted = bool(second_pass_signals.get("adopted"))

    # ------------------------------------------------------------------
    # OCR-confusable micro-repair (plan Phase 5 channels 2+3;
    # SEMANTIK_OCR_CONFUSABLE_REPAIR). Runs on the CONVERGED post-cap/verify
    # ``capped`` regions (text is FB-derived, so this covers both fast/offline
    # lanes) and BEFORE Stage 13. It NEVER mutates region text: it emits an
    # ADDITIVE per-region edits map (threaded into region_provenance) + a
    # repair-stats object that amends the theta exit signal. OFF -> byte-
    # identical: no detector work, no LLM call, no ``ocr_repair`` result key.
    # ------------------------------------------------------------------
    ocr_repair_result = None
    from .qwen_specialists.ocr_repair import resolve_ocr_confusable_repair_mode

    if resolve_ocr_confusable_repair_mode():
        from .qwen_specialists.ocr_repair import run_ocr_confusable_repair
        from .qwen_specialists.reviewer import resolve_structure_review_temperature
        from .qwen_specialists.runtime import (
            make_runtime,
            resolve_structure_review_model,
        )

        log("[cascade] running OCR-confusable micro-repair pass")
        t = time.perf_counter()
        # Rides the SAME already-licensed hosted/ollama specialist seat as the
        # Stage-5d reviewer (make_runtime('endpoint'), the reviewer model pin).
        repair_runtime = make_runtime(
            "endpoint", model=resolve_structure_review_model()
        )
        ocr_repair_result = run_ocr_confusable_repair(
            list(capped),
            feature_blocks,
            repair_runtime,
            temperature=resolve_structure_review_temperature(),
            log=log,
        )
        stages["ocr_repair"] = time.perf_counter() - t

    # OLLAMA LEASE #2 — hand the card off from the Stage-9 second-pass verifier
    # + the SEMANTIK_OCR_CONFUSABLE_REPAIR pass (both ride the SAME ollama
    # specialist seat) AFTER ocr_repair completes and BEFORE Stage 13. Covering
    # both same-seat consumers in ONE lease window avoids a cold 7B reload
    # between them — the repair pass runs post-verify-loop, so this is the
    # correct end-of-window seam. No-op unless a model was resident; gated on
    # ED4ALL_GPU_LIFECYCLE, fail-soft. (Stage 13's offline retry may re-enter
    # _run_inner → Stages 6-12, which reload lazily.)
    _gpu_lifecycle_release(ollama=True, stage="post-second-pass+ocr_repair")

    # Stage 13 — orchestrate the offline retry first, THEN stamp.
    offline_retry_fired = False

    def _run_lane(lane: str) -> Any:
        nonlocal offline_retry_fired
        if lane == "offline":
            offline_retry_fired = True
        return _run_inner(lane)

    # GRACEFUL-STOP SEAM (b) — pre-Stage-13 offline retry. The fast lane's
    # Stages 6-12 already ran; the offline retry may re-run all of Stage 6-12
    # against the ``offline`` lane (another full authoring pass). If a stop was
    # requested during the fast lane, raise here so we don't spend the offline
    # retry on a run we're about to pause. No-op / fail-soft exactly like seam
    # (a) when no SEMANTIK_STOP_SENTINEL was handed in.
    check_cascade_stop("cascade:pre-stage13-offline-retry")

    log("[cascade] running Stage 13 (offline retry orchestration)")
    t = time.perf_counter()
    final_report = maybe_offline_retry(fast_report, run_lane=_run_lane)
    # Channel 3: amend the FINAL report's stubbed semantic-preservation with the
    # repair-stats proxy (no-op when theta is real / stats absent). Applied AFTER
    # maybe_offline_retry so the repair score informs the exit STAMP but does NOT
    # gate the offline retry (documented in theta/exits.py). Byte-stable off.
    if ocr_repair_result is not None:
        final_report = apply_repair_stats(final_report, ocr_repair_result.stats)
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

    # P3a (SemantiK migration §3.3a/§3.5) — distill the already-computed
    # per-region provenance the Ed4All adapter IR needs. ``region_order``
    # is the assembler's ``region_provenance`` (emitted-block → capped-
    # region index, in document order); fall back to the natural region
    # order when the assembler did not surface it (mock / partial runs).
    region_order = list(getattr(assembled, "region_provenance", None) or [])
    if not region_order:
        region_order = list(range(len(capped)))
    region_provenance = _build_region_provenance(
        region_order,
        list(capped),
        feature_blocks,
        chosen["stage7_results"],
        # ITEM4: the derived containment forest the FINAL assembly walked
        # (stashed by pass_9a; None when SEMANTIK_CONTAINMENT is off or the
        # assembler did not surface it). Additive provenance key only.
        containment_tree=(getattr(assembled, "sub_task_log", None) or {}).get(
            "containment"
        ),
        review_verdicts=review_verdicts,
        # Phase 5: the pre-5e region list the verdict block_ids index into, so
        # the verdict->provenance join re-keys by stable min-FB and survives a
        # Stage-5e N->1 regroup. Byte-stable when no merge fires.
        #
        # Phase 6 drift-fix (RECONCILIATION DELTA): when the verify-refine loop
        # ADOPTED a re-typed round, ``review_verdicts`` came from
        # ``run_structure_review(capped, ...)`` so its block_ids are in CAPPED
        # space (NOT the pre-5e ``pre_review_regions`` space). Joining a capped
        # block_id against the pre-5e list would corrupt the min-FB re-key, so
        # the adopted path uses ``review_regions=None`` (the clean positional /
        # capped-consistent join). On a clean / reverted / no-adopt run the
        # verdicts are still the Pass-1 (pre-5e) snapshot -> keep
        # ``pre_review_regions``.
        review_regions=(None if second_pass_adopted else pre_review_regions),
        # OCR-confusable repair edits (keyed by region_index; None when the pass
        # is off / made no accepted edit). Additive → byte-stable when absent.
        ocr_repair_edits=(
            ocr_repair_result.edits_by_region if ocr_repair_result else None
        ),
    )

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

    # Stage-5d structure-review audit section — verdicts-as-dicts when the
    # reviewer ran, None when off (the None-vs-[] reviewer-did-not-run vs
    # ran-found-nothing distinction). dataclasses.asdict serializes each
    # frozen ReviewVerdict; build_conformance_audit treats it as OPTIONAL.
    # Phase 2 (SEMANTIK_BLOCK_REVIEW) byte-stability (F1): the new
    # ``ReviewVerdict.role_after`` field is Optional-default-None and is
    # EXCLUDED from the audit row when None, so a flag-off / heading-only run's
    # audit dict is byte-identical to today (asdict would otherwise emit
    # ``role_after: null`` on every verdict, including the shipped heading
    # path). The exclusion is SCOPED to ``role_after`` ONLY — a blanket
    # None-key drop would strip the legitimately-None ``level_before`` /
    # ``level_after`` keys that today's audit already emits for non-heading
    # regions, which would itself break byte-stability.
    def _verdict_audit_row(verdict: Any) -> dict[str, Any]:
        row = dataclasses.asdict(verdict)
        if row.get("role_after") is None:
            row.pop("role_after", None)
        # Phase 5 (SEMANTIK_BLOCK_REVIEW) byte-stability: the per-window
        # ``block_review_window`` capture-metadata field is Optional-default-None
        # and EXCLUDED when None (SCOPED, exactly like ``role_after`` — never a
        # blanket None-key drop, which would strip the legitimately-None
        # ``level_before`` / ``level_after`` keys). A flag-off / heading-only run
        # never populates it, so its audit dict stays byte-identical to today.
        if row.get("block_review_window") is None:
            row.pop("block_review_window", None)
        # Phase 4 (SEMANTIK_SEMANTIC_CLASS) byte-stability: the audit-only
        # ``semantic_class_after`` field is Optional-default-None and EXCLUDED
        # when None (SCOPED, exactly like ``role_after`` / ``block_review_window``
        # — never a blanket None-key drop, which would strip the legitimately-None
        # ``level_before`` / ``level_after`` keys). A flag-off / heading-only run
        # never populates it, so its audit dict stays byte-identical to today.
        if row.get("semantic_class_after") is None:
            row.pop("semantic_class_after", None)
        return row

    structure_review_audit = (
        [_verdict_audit_row(v) for v in review_verdicts]
        if review_verdicts is not None
        else None
    )

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
        structure_review=structure_review_audit,
        # OCR-confusable repair audit (None when the pass is off → key absent,
        # byte-stable). Parallel to the structure_review / second_pass arms.
        ocr_repair=(ocr_repair_result.stats if ocr_repair_result else None),
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
        # P3a — distilled per-region provenance in document order (the
        # adapter-IR source). Stable, JSON-safe; consumed in-process by
        # ``lib/semantik/cascade_ir.py::build_chapters_ir``.
        "region_provenance": region_provenance,
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
    # Stage-5e block join/split audit section — emitted ONLY when the pass
    # ran (default OFF -> the key is absent, so the result dict is byte-stable
    # to a no-Stage-5e run). Parallel to the structure_review audit.
    if resegment_ops_audit is not None:
        result["block_resegment"] = resegment_ops_audit
    # Stage 9-verify (Phase 6) audit section — emitted ONLY when the verify-
    # refine loop actually ran (SEMANTIK_SECOND_PASS on AND a Pass-1 reviewer
    # produced verdicts AND >=1 round executed). Default OFF -> the key is
    # absent, so the result dict is byte-stable to a no-second-pass run.
    # Parallel to the structure_review / block_resegment audit sections; the
    # bridge (Phase 7) resolves the per-round capture off this arm.
    if second_pass_signals is not None and second_pass_signals.get("rounds"):
        result["second_pass_verify"] = second_pass_signals
    # OCR-confusable repair audit arm — emitted ONLY when the pass ran (default
    # OFF → key absent, byte-stable). Parallel to structure_review /
    # block_resegment / second_pass_verify; the bridge reads it for the capture.
    if ocr_repair_result is not None and ocr_repair_result.stats:
        result["ocr_repair"] = ocr_repair_result.stats
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
    # P3a — distilled per-region provenance in document order (mirrors
    # ``cascade["region_provenance"]``; promoted to a top-level field so the
    # Ed4All seam consumes it without reaching into the telemetry dict).
    region_provenance: list[dict] = dataclasses.field(default_factory=list)
    heading_tree: list[tuple[int, str]] = dataclasses.field(default_factory=list)


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
        # A hosted endpoint on EITHER Stage-6 phase seat IS real generation,
        # so it forces "real" even when no LOCAL GGUF adapter is configured —
        # the per-phase runtime then resolves the endpoint runtime and skips
        # the local-weight presence check for that phase. Only the Stage-6
        # specialists are affected; council/theta stay local.
        if any_phase_provider_is_endpoint() or config.is_qwen_specialist_loaded():
            runtime_mode = "real"
        else:
            runtime_mode = "mock"

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
        region_provenance=list(result.get("region_provenance") or []),
        heading_tree=[tuple(t) for t in (result.get("heading_tree") or [])],
    )


__all__ = [
    "run_full_cascade",
    "run_pipeline_v2",
    "PipelineV2Result",
]
