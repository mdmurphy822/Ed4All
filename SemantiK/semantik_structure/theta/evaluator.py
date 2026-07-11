"""Stage 12 — ThetaEvaluator (post-WCAG meaning-preservation diagnostic).

Composites 1 learned + 7 deterministic dimensions over an
:class:`AssembledDoc` and emits a :class:`ThetaReport`. v1 implements 7
dimensions deterministically; ``semantic_preservation`` is **model-backed
when the Phase 4b cross-encoder is on disk and passes the sentinel
health check** (see :mod:`semantik_structure.theta.semantic_preservation`).

**Strict-by-default**: when the model is missing, fails to load, or
fails the sentinel, ``evaluate()`` raises ``RuntimeError`` instead of
silently falling back to the 0.7 ``stub_v1`` placeholder. The stub
remains opt-in via ``DART_ALLOW_THETA_STUB=1``; see
:mod:`semantik_structure.theta._module_state` for the policy. The stub-only
path stays import-light: no torch / transformers / peft are imported
when running stub-only.

The composite formula loads its per-dimension weights from
``theta/config.yaml`` (:func:`semantik_structure.theta.types.load_theta_config`,
cached). Since theta-config-2.0 those weights — and the TAU_* exit
thresholds + per-dimension floors — are CALIBRATED against synthetic
perturbations of WCAG-clean pair HTML (``scripts/calibrate_theta.py``;
provenance in the config's ``calibration:`` block). A missing or
invalid config raises :class:`~semantik_structure.theta.types.ThetaConfigError`
— there is no silent fall-back to uniform weights.

All ``_score_*`` helpers are pure functions of their inputs and degrade
to a neutral 1.0 when the underlying signal is absent (e.g. zero
internal anchors → ``ref_integrity = 1.0``). Each clamps its output
to ``[0.0, 1.0]``. They do not raise.

Stage 13 (:func:`semantik_structure.theta.exits.decide_exit`) reads the
returned :class:`ThetaReport` and stamps ``action``; ``evaluate``
itself leaves ``action`` set to the v1 stub
:attr:`ConfidenceAction.SHIP_WITH_CONFIDENCE` so the report is
self-consistent if Stage 13 is skipped.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Literal

from semantik_structure.assembler.types import AssembledDoc, GapKind
from semantik_structure.soft_reranker.document import (
    _score_heading_tree_balance,
    _score_ref_link_integrity,
)
from semantik_structure.soft_reranker.types import ScoredDoc
from semantik_structure.types import FeatureBlock

from .types import (
    CognitiveLoadRisk,
    ConfidenceAction,
    DimensionScore,
    ThetaDimension,
    ThetaFlag,
    ThetaReport,
    floors_from_config,
    load_theta_config,
)


# ---------------------------------------------------------------------------
# Module-level regexes (compiled once).
# ---------------------------------------------------------------------------
_HREF_HASH_RE = re.compile(r'<a\s+[^>]*href="#([^"]+)"', re.IGNORECASE)
_ID_ATTR_RE = re.compile(r'\bid="([^"]+)"', re.IGNORECASE)
_HEADING_RE = re.compile(r"<(h[1-6])(\s[^>]*)?>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_HEADING_HAS_ID_RE = re.compile(r'<h[1-6]\s+[^>]*id="[^"]+"', re.IGNORECASE)
_PARA_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_LIST_RE = re.compile(r"<(ul|ol|dl)\b", re.IGNORECASE)
_NAV_TOC_RE = re.compile(r"<nav\b[^>]*>(.*?)</nav>", re.IGNORECASE | re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
# Marker emitted by the table specialist runtime (mock + real adapter
# prompt template) when ``cell_roles`` was None at prompt-build time —
# i.e. BERT-TableSpecialist didn't supply per-cell header signals and
# Qwen-Table inferred them on its own. See
# ``semantik_structure.qwen_specialists.prompts.build_table_request``.
_TABLE_QWEN_INFERRED_RE = re.compile(
    r'data-dart-cell-roles="qwen-inferred"',
    re.IGNORECASE,
)

#: Canonical ``breakdown["method"]`` value the semantic-preservation
#: dimension carries when the cross-encoder could not be loaded and the
#: 0.7 placeholder was substituted (``DART_ALLOW_THETA_STUB=1``). This is
#: the single source of truth for the stub marker — ``theta_is_stubbed``
#: keys off it, so a stubbed run is detectable from the ``ThetaReport``
#: alone (no env re-read needed at the Stage-13 seam).
STUB_SEMANTIC_METHOD: str = "stub_v1"


#: Canonical ``breakdown["method"]`` value the semantic-preservation dimension
#: carries when the OCR-confusable repair-stats proxy (channel 3) REPLACED the
#: stub_v1 placeholder. Distinct from a real cross-encoder method string and
#: from STUB_SEMANTIC_METHOD, so an amended report is self-describing.
OCR_REPAIR_STATS_METHOD: str = "ocr_repair_stats_v1"


def apply_repair_stats(report: ThetaReport, stats: dict | None) -> ThetaReport:
    """Amend a STUBBED theta report's semantic-preservation with repair stats.

    Channel 3 of the scan-conversion plan: replace the flat-0.7 ``stub_v1``
    placeholder with the OCR-confusable pass's ``repair_stats_score`` (an honest
    residual-garble-density proxy, ceilinged < 1.0), recompute the composite with
    the SAME cached ``load_theta_config()`` weights, and re-derive the
    ``MEANING_PRESERVATION_LOW`` floor flag against the config floor.

    Fires ONLY when BOTH hold: :func:`theta_is_stubbed` is True (a real-model run
    is NEVER overridden) AND ``stats`` carries a ``repair_stats_score``. Any
    other case returns ``report`` UNCHANGED (byte-stable) — so a flag-off /
    no-stats / real-model run is untouched. The amended dimension's method is
    :data:`OCR_REPAIR_STATS_METHOD`, so ``theta_is_stubbed`` (which keys off
    ``stub_v1`` only) reports the amended report as NO LONGER stubbed → the
    Stage-13 exit-decider naturally takes the real tau threshold path.
    """
    if not theta_is_stubbed(report):
        return report
    if not isinstance(stats, dict):
        return report
    raw_score = stats.get("repair_stats_score")
    if raw_score is None:
        return report
    try:
        repair_score = float(raw_score)
    except (TypeError, ValueError):
        return report

    dims = dict(report.dimensions or {})
    sem_key = ThetaDimension.SEMANTIC_PRESERVATION.value
    sem = dims.get(sem_key)
    if not isinstance(sem, dict):
        return report

    new_sem = dict(sem)
    new_sem["score"] = _clamp(repair_score)
    new_sem["breakdown"] = {
        "method": OCR_REPAIR_STATS_METHOD,
        "unrepairable_defect_density": stats.get("unrepairable_defect_density"),
        "flagged": stats.get("flagged_blocks"),
        "accepted": stats.get("accepted"),
    }
    dims[sem_key] = new_sem

    # Recompute the composite with the SAME calibrated weights.
    cfg = load_theta_config()
    weights = cfg.weights
    theta_score = 0.0
    for name, d in dims.items():
        score = d.get("score", 0.0) if isinstance(d, dict) else 0.0
        theta_score += weights.get(name, 0.0) * float(score)
    theta_score = round(_clamp(theta_score), 4)

    # Re-derive the MEANING_PRESERVATION_LOW floor flag (drop a stale one, add
    # it back only when the repaired score is below the config floor).
    floors = floors_from_config(cfg)
    sem_floor = floors.get("semantic_preservation")
    flags = [f for f in (report.flags or []) if f != ThetaFlag.MEANING_PRESERVATION_LOW.value]
    if sem_floor is not None and _clamp(repair_score) < sem_floor:
        flags.append(ThetaFlag.MEANING_PRESERVATION_LOW.value)

    return dataclasses.replace(
        report, dimensions=dims, theta_score=theta_score, flags=flags
    )


def theta_is_stubbed(report: ThetaReport | None) -> bool:
    """True iff this report's semantic_preservation dimension was STUBBED.

    The theta cross-encoder (v8) is mode-collapsed on this box, so the
    ``DART_ALLOW_THETA_STUB=1`` path substitutes a flat 0.7 placeholder
    for the learned ``semantic_preservation`` dimension. That makes the
    composite ``theta_score`` meaningless — it MUST NOT gate the offline
    retry or the exit decision (see ``offline_retry._needs_retry`` /
    ``exits.decide_exit``).

    Detection keys ONLY off the report's own
    ``dimensions["semantic_preservation"]["breakdown"]["method"]`` ==
    :data:`STUB_SEMANTIC_METHOD` — the marker the evaluator stamps when
    ``_get_model()`` returned ``None``. No env re-read: the report is
    self-describing, so a report serialized to JSON and re-read at any
    seam yields the same verdict. Returns ``False`` for a None report,
    an empty/failed-WCAG report (no dimensions), or a real-model run
    (byte-stable: the real path's breakdown carries the model's scoring
    method, never ``stub_v1``).
    """
    if report is None:
        return False
    dims = getattr(report, "dimensions", None) or {}
    sem = dims.get(ThetaDimension.SEMANTIC_PRESERVATION.value)
    if not isinstance(sem, dict):
        return False
    breakdown = sem.get("breakdown")
    if not isinstance(breakdown, dict):
        return False
    return breakdown.get("method") == STUB_SEMANTIC_METHOD


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def evaluate(
    doc: AssembledDoc,
    *,
    feature_blocks: Sequence[FeatureBlock],
    regions: Sequence | None = None,
    wcag_status: Literal["passed", "failed"] = "passed",
    lane: Literal["fast", "offline"] = "fast",
    stage11_scored: ScoredDoc | None = None,
    theta_version: str | None = None,
    reading_order_at_risk: Sequence[int] = (),
) -> ThetaReport:
    """Evaluate the 8 theta dimensions over an assembled doc.

    On ``wcag_status == "failed"`` returns an empty report with
    ``theta_score = None`` (not 0.0 — we did not measure) and no
    dimensions populated; Stage 13 is responsible for the final action
    and will raise :class:`StageThirteenStubRequired` for the v1-stub
    "fail | fast" row.

    Otherwise runs all 8 dimensions, composites them with the
    calibrated per-dimension weights from ``theta/config.yaml``
    (cached load; raises ``ThetaConfigError`` if the config is missing
    or invalid — never a silent uniform fallback), applies the
    per-dimension floors from the same config, and returns a
    :class:`ThetaReport`. ``theta_version`` defaults to the loaded
    config's version. ``action`` is provisional —
    :func:`semantik_structure.theta.exits.decide_exit` overwrites it with the
    Stage-13 verdict.
    """
    cfg = load_theta_config()
    if theta_version is None:
        theta_version = cfg.version
    if wcag_status == "failed":
        # Per architecture.md §6 / §7.4: theta is omitted on failed-WCAG.
        # theta_score=None makes "we did not measure" explicit; a 0.0
        # score would silently look like "score was zero" on a dashboard.
        return ThetaReport(
            schema_version="theta/1.0",
            wcag_status="failed",
            lane=lane,
            theta_score=None,
            theta_version=theta_version,
            dimensions={},
            flags=[],
            action=ConfidenceAction.NON_CERTIFIED_STAMP,
        )

    dim_scores: dict = {}

    # 1. semantic_preservation — model-backed when the Phase 4b
    # cross-encoder is on disk; documented stub fallback otherwise.
    from ._module_state import _get_model

    _model = _get_model()
    if _model is None:
        sem = DimensionScore(
            dimension=ThetaDimension.SEMANTIC_PRESERVATION,
            score=0.7,
            breakdown={"method": STUB_SEMANTIC_METHOD, "reason": "model_unavailable"},
        )
    else:
        from . import semantic_preservation as _sp

        sp_score, sp_breakdown = _sp.score(
            doc,
            feature_blocks=feature_blocks,
            regions=regions,
            model=_model,
        )
        sem = DimensionScore(
            dimension=ThetaDimension.SEMANTIC_PRESERVATION,
            score=sp_score,
            breakdown=sp_breakdown,
        )
    dim_scores[ThetaDimension.SEMANTIC_PRESERVATION.value] = sem

    # 2. structural_coherence — reuse Stage 11 heading_tree_balance.
    if stage11_scored is not None and "heading_tree_balance" in stage11_scored.breakdown:
        heading_balance = float(stage11_scored.breakdown["heading_tree_balance"])
    else:
        heading_balance = _score_heading_tree_balance(doc)
    dim_scores[ThetaDimension.STRUCTURAL_COHERENCE.value] = DimensionScore(
        dimension=ThetaDimension.STRUCTURAL_COHERENCE,
        score=heading_balance,
        breakdown={"method": "reused:stage11"},
    )

    # 3. navigation_clarity — landmark × heading-id density × ToC-resolution.
    landmark = _score_landmark_coverage(doc)
    heading_id_density = _score_heading_id_density(doc)
    toc_res = _score_toc_resolution(doc)
    nav = (landmark + heading_id_density + toc_res) / 3.0
    dim_scores[ThetaDimension.NAVIGATION_CLARITY.value] = DimensionScore(
        dimension=ThetaDimension.NAVIGATION_CLARITY,
        score=_clamp(nav),
        breakdown={
            "landmark_coverage": landmark,
            "heading_id_density": heading_id_density,
            "toc_resolution": toc_res,
        },
    )

    # 4. context_continuity — section preservation + intra-section refs.
    section_pres = _score_section_preservation(doc, feature_blocks)
    intra_ref = _score_intra_section_ref_res(doc)
    ctx = (section_pres + intra_ref) / 2.0
    dim_scores[ThetaDimension.CONTEXT_CONTINUITY.value] = DimensionScore(
        dimension=ThetaDimension.CONTEXT_CONTINUITY,
        score=_clamp(ctx),
        breakdown={
            "section_preservation": section_pres,
            "intra_section_ref": intra_ref,
        },
    )

    # 5. reference_integrity — reuse Stage 11.
    if stage11_scored is not None and "ref_link_integrity" in stage11_scored.breakdown:
        ref_int = float(stage11_scored.breakdown["ref_link_integrity"])
    else:
        ref_int = _score_ref_link_integrity(doc)
    dim_scores[ThetaDimension.REFERENCE_INTEGRITY.value] = DimensionScore(
        dimension=ThetaDimension.REFERENCE_INTEGRITY,
        score=ref_int,
        breakdown={"method": "reused:stage11"},
    )

    # 6. cognitive_load_reduction — composite + risk enum.
    cog_score, cog_risk = _score_cognitive_load(doc)
    dim_scores[ThetaDimension.COGNITIVE_LOAD_REDUCTION.value] = DimensionScore(
        dimension=ThetaDimension.COGNITIVE_LOAD_REDUCTION,
        score=cog_score,
        breakdown={
            "risk": cog_risk.value,
            "risk_score": {"low": 1.0, "medium": 0.5, "high": 0.0}[cog_risk.value],
        },
    )

    # 7. fragmentation_penalty — inverse of source paragraph split rate.
    frag = _score_fragmentation(doc, feature_blocks)
    dim_scores[ThetaDimension.FRAGMENTATION_PENALTY.value] = DimensionScore(
        dimension=ThetaDimension.FRAGMENTATION_PENALTY,
        score=frag,
    )

    # 8. hallucinated_structure_penalty — gap-fill token anchoring.
    halluc, halluc_breakdown = _score_hallucinated_structure(doc, feature_blocks)
    dim_scores[ThetaDimension.HALLUCINATED_STRUCTURE_PENALTY.value] = DimensionScore(
        dimension=ThetaDimension.HALLUCINATED_STRUCTURE_PENALTY,
        score=halluc,
        breakdown=halluc_breakdown,
    )

    # Composite — calibrated per-dimension weights from theta/config.yaml
    # (uniform 1/8 was the v1 placeholder; replaced by Plan 12 A3).
    weights = cfg.weights
    theta_score = sum(weights[name] * d.score for name, d in dim_scores.items())
    theta_score = round(_clamp(theta_score), 4)

    # Per-dimension floor checks → ThetaFlag list. Floors come from the
    # same config load as the weights so they share calibration provenance.
    flags: list = []
    flag_map = {
        "reference_integrity": ThetaFlag.BROKEN_REFS_PRESENT.value,
        "hallucinated_structure_penalty": ThetaFlag.GAP_FILL_REVIEW_RECOMMENDED.value,
        "semantic_preservation": ThetaFlag.MEANING_PRESERVATION_LOW.value,
    }
    for dim_name, floor in floors_from_config(cfg).items():
        if dim_scores[dim_name].score < floor:
            flags.append(flag_map[dim_name])
    if cog_risk == CognitiveLoadRisk.HIGH:
        flags.append(ThetaFlag.COGNITIVE_LOAD_HIGH.value)

    # Title-fabrication audit (no-silent-fallbacks policy). Observational
    # only — does NOT change the Stage 13 action; surfaced so downstream
    # consumers (eval JSON, sidecar dashboards) can see when a document's
    # <title>/<h1> were synthesised by the assembler rather than extracted
    # from the source. Two paths set this:
    #
    #   * 9c MISSING_TITLE fallback fired (zero gap-fill survivors) →
    #     entry in ``doc.gaps_fallback`` with kind == MISSING_TITLE.
    #   * 9a defaulted with no doc_role:title source AND no input h1 AND
    #     ``skip_gap_fill=True`` (or ``gap_fill_k=0``) → no gap-fill was
    #     attempted, but the title is still synthetic. The literal log
    #     value is set in ``pass_9a.py`` Sub-task 3.
    title_fabricated = False
    for gap in doc.gaps_fallback or []:
        if gap.kind is GapKind.MISSING_TITLE:
            title_fabricated = True
            break
    if not title_fabricated:
        log_val = (doc.sub_task_log or {}).get("title_detect")
        if log_val == "source=default":
            title_fabricated = True
    if title_fabricated:
        flags.append(ThetaFlag.TITLE_FABRICATED.value)

    # Table-headers-inferred audit (no-silent-fallbacks policy).
    # Observational only — does NOT change the Stage 13 action; surfaced
    # so eval JSON / sidecar dashboards can see when at least one
    # ``<table>`` shipped with header semantics inferred by Qwen-Table
    # without a verified BERT-TableSpecialist cell_role signal. The
    # marker is emitted by the table specialist runtime (mock + the
    # real adapter's prompt template) whenever
    # ``region.payload["cell_roles"]`` was ``None`` at prompt-build
    # time. We grep ``doc.html`` directly rather than plumb a separate
    # state field — the marker is the single source of truth, and any
    # downstream rewrite that strips it is itself a regression we want
    # to catch (the audit goes silent, the dashboards see a doc that
    # was never marked best-effort).
    if _TABLE_QWEN_INFERRED_RE.search(doc.html or ""):
        flags.append(ThetaFlag.TABLE_HEADERS_INFERRED.value)

    # Reading-order-at-risk audit (Plan 12 B3, no-silent-fallbacks
    # policy). Observational only — does NOT change the Stage 13 action.
    # The cascade computes which regions fell back to deterministic
    # emission while being/adjoining a complex region (table / figure /
    # math) in reading order; SC 1.3.2 cannot be machine-verified
    # post-hoc, so the degradation is surfaced instead of swallowed.
    if reading_order_at_risk:
        flags.append(ThetaFlag.READING_ORDER_AT_RISK.value)

    # Serialize DimensionScore objects to dicts on the report so JSON
    # round-trip works the same way Phase-0 test_theta_types.py expects.
    dim_dict = {k: dataclasses.asdict(v) for k, v in dim_scores.items()}

    return ThetaReport(
        schema_version="theta/1.0",
        wcag_status="passed",
        lane=lane,
        theta_score=theta_score,
        theta_version=theta_version,
        dimensions=dim_dict,
        flags=flags,
        # Stage 13 fills this in; placeholder is the canonical happy-path
        # value so the report is self-consistent if exits is skipped.
        action=ConfidenceAction.SHIP_WITH_CONFIDENCE,
    )


# ---------------------------------------------------------------------------
# Per-dimension scorers (deterministic).
# ---------------------------------------------------------------------------


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _score_landmark_coverage(doc: AssembledDoc) -> float:
    """Landmark coverage — same shape as Stage 11.

    Required: at least one ``<main>``. Auxiliary landmarks each add
    ``+0.08`` from a base of 0.6, capped at 1.0. v1 simplification:
    we accept whatever ``doc.landmarks`` contains; we don't re-scan
    the HTML.
    """
    landmarks = doc.landmarks or {}
    if landmarks.get("main", 0) < 1:
        return 0.0
    bonus = sum(
        1 for k in ("address", "header", "footer", "nav", "aside") if landmarks.get(k, 0) > 0
    )
    return min(1.0, 0.6 + 0.08 * bonus)


def _score_heading_id_density(doc: AssembledDoc) -> float:
    """Ratio of headings with ``id="…"`` over total headings.

    1.0 when the doc has zero headings (vacuously navigable). The
    regex match is permissive — any ``id=`` attribute on the opening
    tag counts, regardless of value content.
    """
    html = doc.html or ""
    headings = _HEADING_RE.findall(html)
    if not headings:
        return 1.0
    with_id = len(_HEADING_HAS_ID_RE.findall(html))
    return _clamp(with_id / len(headings))


def _score_toc_resolution(doc: AssembledDoc) -> float:
    """If the doc has a ``<nav>`` containing internal anchors, score
    by ratio of those anchors that resolve to in-doc ids. 1.0 when
    no ``<nav>`` block contains any ``href="#…"`` (vacuous).

    v1 simplification: we treat *every* ``<nav>`` block as a ToC. A
    fancier v2 would gate by ``aria-label`` or first heading text.
    """
    html = doc.html or ""
    nav_blocks = _NAV_TOC_RE.findall(html)
    if not nav_blocks:
        return 1.0
    nav_html = "\n".join(nav_blocks)
    refs = _HREF_HASH_RE.findall(nav_html)
    if not refs:
        return 1.0
    ids = set(_ID_ATTR_RE.findall(html))
    resolved = sum(1 for r in refs if r in ids)
    return _clamp(resolved / len(refs))


def _score_section_preservation(
    doc: AssembledDoc,
    feature_blocks: Sequence[FeatureBlock],
) -> float:
    """Ratio of source paragraph 'sections' that map to a single
    output region vs being split across multiple.

    v1 simplification: we don't have a section-id mapping in the
    AssembledDoc. We approximate by comparing the count of
    consecutive prose runs in the source feature_blocks to the
    count of ``<section>``/heading-anchored regions in the output.
    Returns 1.0 if either count is zero (cannot measure).
    """
    if not feature_blocks:
        return 1.0
    # Count source "sections": runs of feature_blocks separated by
    # large vertical gaps or font-size-bucket increases.
    src_sections = 0
    last_size = None
    for fb in feature_blocks:
        if fb.gap_above == "lg" or (
            last_size and fb.size_bucket != last_size and fb.size_bucket in {"xl", "lg"}
        ):
            src_sections += 1
        last_size = fb.size_bucket
    src_sections = max(1, src_sections)

    html = doc.html or ""
    out_headings = len(_HEADING_RE.findall(html))
    if out_headings == 0:
        return 1.0
    # When output has at least as many heading anchors as source
    # sections, treat sections as preserved. Otherwise score by
    # ratio of headings to sections (capped to 1.0).
    return _clamp(out_headings / src_sections)


def _score_intra_section_ref_res(doc: AssembledDoc) -> float:
    """Resolution rate of intra-doc anchors. v1 simplification: same
    as ref_link_integrity but counted only when at least one heading
    with an id exists (proxy for "the doc has sections to link").
    Returns 1.0 if no intra-doc refs exist.
    """
    html = doc.html or ""
    refs = _HREF_HASH_RE.findall(html)
    if not refs:
        return 1.0
    ids = set(_ID_ATTR_RE.findall(html))
    resolved = sum(1 for r in refs if r in ids)
    return _clamp(resolved / len(refs))


def _score_cognitive_load(doc: AssembledDoc) -> tuple[float, CognitiveLoadRisk]:
    """Composite of paragraph-length distribution, heading density,
    and list coverage. Emits a :class:`CognitiveLoadRisk` enum.

    Components (each in [0, 1], higher = better):
      * para_len_score: 1.0 if mean para word count ≤ 80, decays
        linearly to 0.0 at mean ≥ 200 words.
      * heading_density: 1.0 when 1 heading per 5–15 paragraphs;
        decays outside that band.
      * list_coverage: 1.0 if doc has at least one list element,
        else 0.5 (neutral; lists not always appropriate).

    Risk:
      * composite ≥ 0.7  → low
      * composite ≥ 0.4  → medium
      * else             → high
    """
    html = doc.html or ""
    para_texts = [_TAG_STRIP_RE.sub("", p).strip() for p in _PARA_RE.findall(html)]
    para_count = len(para_texts)
    if para_count == 0:
        # No paragraphs — neutral score, low risk (likely a small doc).
        return 0.7, CognitiveLoadRisk.LOW

    # Paragraph length score.
    word_counts = [len(t.split()) for t in para_texts if t]
    if word_counts:
        mean_words = sum(word_counts) / len(word_counts)
    else:
        mean_words = 0.0
    if mean_words <= 80:
        para_len_score = 1.0
    elif mean_words >= 200:
        para_len_score = 0.0
    else:
        para_len_score = 1.0 - (mean_words - 80) / (200 - 80)

    # Heading density.
    heading_count = len(_HEADING_RE.findall(html))
    if heading_count == 0:
        heading_density_score = 0.3
    else:
        ratio = heading_count / max(1, para_count)
        if 1 / 15 <= ratio <= 1 / 5:
            heading_density_score = 1.0
        elif ratio < 1 / 15:
            heading_density_score = ratio / (1 / 15)
        else:  # ratio > 1/5 (too many headings)
            heading_density_score = max(0.3, 1.0 - (ratio - 1 / 5))

    # List coverage.
    list_coverage = 1.0 if _LIST_RE.search(html) else 0.5

    composite = (para_len_score + heading_density_score + list_coverage) / 3.0
    composite = _clamp(composite)

    if composite >= 0.7:
        risk = CognitiveLoadRisk.LOW
    elif composite >= 0.4:
        risk = CognitiveLoadRisk.MEDIUM
    else:
        risk = CognitiveLoadRisk.HIGH
    return composite, risk


def _score_fragmentation(
    doc: AssembledDoc,
    feature_blocks: Sequence[FeatureBlock],
) -> float:
    """Inverse of source-paragraph-split rate.

    v1 simplification: we estimate "source paragraphs" as the count
    of feature_blocks whose ``size_bucket`` is ``"md"`` or smaller
    AND whose text contains at least 5 words (filters out heading-
    sized blocks and very short fragments). Then compare to the
    number of ``<p>`` tags in the output. A ratio of output-paragraphs
    to source-paragraphs > 1.5 signals splitting; the score decays
    linearly from 1.0 at ratio ≤ 1.0 to 0.0 at ratio ≥ 3.0.

    Returns 1.0 if either count is zero (cannot measure).
    """
    html = doc.html or ""
    out_paras = len(_PARA_RE.findall(html))
    if out_paras == 0 or not feature_blocks:
        return 1.0
    src_paras = sum(
        1
        for fb in feature_blocks
        if fb.size_bucket in {"md", "sm"} and len((fb.raw.text or "").split()) >= 5
    )
    if src_paras == 0:
        return 1.0
    ratio = out_paras / src_paras
    if ratio <= 1.0:
        return 1.0
    if ratio >= 3.0:
        return 0.0
    return _clamp(1.0 - (ratio - 1.0) / 2.0)


_TAG_RE = re.compile(r"<[^>]+>")
_ANCHOR_TOKEN_RE = re.compile(r"[a-zA-Z]{4,}|\d[\w.,%-]*")


def _anchor_tokens(text: str) -> set[str]:
    """Content tokens for anchoring: words (≥4 alpha chars, lowercased)
    plus ANY numeric token — fabricated numbers are exactly the
    hallucination risk the captioner work surfaced."""
    return {t.lower() for t in _ANCHOR_TOKEN_RE.findall(text or "")}


def _flatten_strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _flatten_strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _flatten_strings(v)]
    return []


def _score_hallucinated_structure(
    doc: AssembledDoc,
    feature_blocks: Sequence[FeatureBlock],
) -> tuple[float, dict]:
    """Penalize gap-fill-emitted text that has no anchor in source.

    Token-level anchoring (Plan 12 A3, replaces the constant-0.85
    v1 rule): for each resolved gap, the visible text of the spliced
    candidate (``GapSlot.resolved_html``, recorded by pass 9c) is
    tokenized into content tokens (words ≥4 alpha chars + ALL numeric
    tokens) and checked for membership in the anchor pool — the gap's
    own extraction context plus the full source text from the feature
    blocks. The dimension score is the WORST per-gap anchored ratio
    (a single fabricated block is the risk, not the average).

    * no resolved gaps → 1.0 (nothing generated, nothing to anchor).
    * a resolved gap with no recorded provenance (legacy caller that
      predates ``resolved_html``) scores the old conservative 0.85 —
      "unmeasured", never "verified".
    * a gap whose text yields zero content tokens (pure markup) → 1.0.
    """
    gaps_resolved = doc.gaps_resolved or []
    if not gaps_resolved:
        return 1.0, {"method": "token_anchoring_v2", "n_gaps": 0}

    source_pool: set[str] = set()
    for fb in feature_blocks:
        source_pool |= _anchor_tokens(getattr(fb.raw, "text", "") or "")

    per_gap: list[dict] = []
    worst = 1.0
    for gap in gaps_resolved:
        if gap.resolved_html is None:
            per_gap.append(
                {
                    "kind": getattr(gap.kind, "value", str(gap.kind)),
                    "score": 0.85,
                    "reason": "no_provenance_recorded",
                }
            )
            worst = min(worst, 0.85)
            continue
        visible = _TAG_RE.sub(" ", gap.resolved_html)
        tokens = _anchor_tokens(visible)
        if not tokens:
            per_gap.append(
                {
                    "kind": getattr(gap.kind, "value", str(gap.kind)),
                    "score": 1.0,
                    "reason": "no_content_tokens",
                }
            )
            continue
        pool = source_pool.copy()
        for s in _flatten_strings(gap.context or {}):
            pool |= _anchor_tokens(s)
        anchored = len(tokens & pool)
        ratio = anchored / len(tokens)
        per_gap.append(
            {
                "kind": getattr(gap.kind, "value", str(gap.kind)),
                "score": round(ratio, 4),
                "n_tokens": len(tokens),
                "n_unanchored": len(tokens) - anchored,
            }
        )
        worst = min(worst, ratio)

    return round(_clamp(worst), 4), {
        "method": "token_anchoring_v2",
        "n_gaps": len(gaps_resolved),
        "per_gap": per_gap,
    }


__all__ = [
    "evaluate",
    "theta_is_stubbed",
    "STUB_SEMANTIC_METHOD",
    "OCR_REPAIR_STATS_METHOD",
    "apply_repair_stats",
]
