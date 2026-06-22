"""End-to-end orchestrator for the 8-stage pipeline.

Contract:
    run_pipeline(pdf_path, *, language=None, classifier=None) -> PipelineResult

Runs stages 1→8 sequentially. Each stage gets a fresh view of the blocks
and annotates them; the result carries HTML, validation report, and the
escalation decision.

Stage 3's classifier argument is optional: when None, the rule-set inside
classify.py runs alone (ambiguous blocks default to PARAGRAPH). Once the
stage-3 LoRA adapter is trained, pass it here and the rules + model
combine.

This orchestrator is intentionally short — the heavy lifting lives in
each stage's module. Debug a failing document by running stages one at
a time and inspecting intermediate block lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import classify, enrich, escalate, hierarchy, ontology_map, reason, validate
from .extract import blocks_from_shared
from .extract_shared import extract_shared
from .features import featurize_from_shared
from .types import ClassifiedBlock, FeatureBlock, RawBlock, ResolvedBlock


@dataclass
class PipelineResult:
    pdf_path: Path
    html: str | None
    raw_blocks: list[RawBlock] = field(default_factory=list)
    feature_blocks: list[FeatureBlock] = field(default_factory=list)
    classified_blocks: list[ClassifiedBlock] = field(default_factory=list)
    resolved_blocks: list[ResolvedBlock] = field(default_factory=list)
    validation: "validate.ValidationResult | None" = None
    escalation: "escalate.EscalationDecision | None" = None
    # Stage-level diagnostics
    stage_errors: list[tuple[int, str]] = field(default_factory=list)


def run_pipeline(pdf_path: Path,
                 *,
                 language: str = "en",
                 classifier=None,
                 reasoner=None,
                 glm_ocr=None,
                 validator: "validate.HtmlValidator | None" = None) -> PipelineResult:
    """Run stages 1-8 on a PDF.

    `classifier` — dict from classify.load_classifier(path). If None, rules
                   fire and ambiguous blocks default to PARAGRAPH.
    `reasoner`   — dict from reason.load_reasoner(path). If None, stage 3b
                   is skipped; distilbert's labels pass through unchanged.
    `glm_ocr`    — dict from glm_ocr.load_glm_ocr(). If provided, stage 1
                   is augmented with region-targeted OCR on every detected
                   math / table region, plus full-page OCR for any page
                   that had to fall back to Tesseract. Results cached to
                   disk so re-runs are free.

    For batch mode, load the classifier once + reuse across documents. If
    reasoner is also loaded at the same time they co-tenant (~3.8 GB total
    VRAM). For live single-document calls, prefer loading/unloading
    sequentially (distilbert → release → reasoner → release).
    """
    result = PipelineResult(pdf_path=pdf_path, html=None)

    # Stage 1: extract (shared multi-extractor JSON)
    try:
        shared = extract_shared(pdf_path)
        if glm_ocr is not None:
            from .glm_ocr_enrich import enrich_shared
            enrich_shared(shared, pdf_path, glm_ocr)
        raw = blocks_from_shared(shared)
    except Exception as exc:
        result.stage_errors.append((1, f"extract: {exc}"))
        return result
    result.raw_blocks = raw
    if not raw:
        result.stage_errors.append((1, "no blocks extracted from PDF"))
        return result

    # Stage 2: features — enriched with pdfplumber tables + pikepdf widgets.
    # FeatureBlock fields that the v1 classifier doesn't know about default
    # to False/None and get skipped by the serializer, so v1 inputs stay
    # identical. v2 classifier trained on the richer format uses them.
    try:
        features = featurize_from_shared(shared)
    except Exception as exc:
        result.stage_errors.append((2, f"featurize: {exc}"))
        return result
    result.feature_blocks = features

    # Stage 3a: classify (distilbert)
    try:
        classified = classify.classify_blocks(features, model=classifier)
    except Exception as exc:
        result.stage_errors.append((3, f"classify: {exc}"))
        return result

    # Stage 3b: Qwen 3 document-level reasoning (optional).
    # Receives distilbert's labels + confidences and emits corrections
    # on blocks where cross-block context reveals a mislabel.
    if reasoner is not None and classified:
        try:
            classified = reason.reason_over_document(classified, reasoner)
        except Exception as exc:
            # Soft-fail: stage 3b is an augmentation, not a requirement.
            # distilbert's labels remain intact if Qwen crashes.
            result.stage_errors.append(
                (3, f"reasoner soft-fail (distilbert labels retained): {exc}"))

    result.classified_blocks = classified

    # Stage 4: hierarchy
    try:
        resolved = hierarchy.resolve_hierarchy(classified)
    except Exception as exc:
        result.stage_errors.append((4, f"hierarchy: {exc}"))
        return result
    result.resolved_blocks = resolved

    # Stage 6: enrich (runs before stage 5 so enrichments can feed into HTML)
    try:
        resolved = enrich.enrich(resolved)
    except Exception as exc:
        result.stage_errors.append((6, f"enrich: {exc}"))
        # continue — enrichment is best-effort

    # Stage 5: ontology map -> HTML
    try:
        html = ontology_map.emit_html(resolved, language=language)
    except ontology_map.MappingError as exc:
        result.stage_errors.append((5, f"mapping: {exc}"))
        return result
    except Exception as exc:
        result.stage_errors.append((5, f"ontology_map unexpected: {exc}"))
        return result
    result.html = html

    # Stage 7: validate
    own_validator = validator is None
    try:
        if own_validator:
            validator = validate.HtmlValidator().__enter__()
        val_result = validator.check(html)
    except Exception as exc:
        result.stage_errors.append((7, f"validate: {exc}"))
        val_result = validate.ValidationResult(ok=False, error=str(exc))
    finally:
        if own_validator and validator is not None:
            try:
                validator.__exit__(None, None, None)
            except Exception:
                pass
    result.validation = val_result

    # Stage 8: escalate
    try:
        n_serious = sum(1 for v in val_result.violations if v.impact == "serious")
        n_critical = sum(1 for v in val_result.violations if v.impact == "critical")
        mean_conf = (sum(cb.confidence for cb in classified) / len(classified)
                     if classified else 0.0)
        claude_fallbacks = sum(
            1 for r in resolved
            if r.enrichment and r.enrichment.alt_text_source == "claude_sonnet_v4"
        )
        page_count = max((b.page for b in raw), default=1)
        # Qwen override rate — only meaningful when reason.py ran. The
        # "qwen3:override" source marker is set by reason_over_document.
        qwen_tags = [cb.source for cb in classified
                     if cb.source.startswith("qwen3:")]
        if qwen_tags:
            n_override = sum(1 for s in qwen_tags if s == "qwen3:override")
            qwen_override_rate = n_override / len(qwen_tags)
        else:
            qwen_override_rate = None
        report = escalate.DocumentReport(
            n_blocks=len(resolved),
            n_axe_serious=n_serious,
            n_axe_critical=n_critical,
            mean_classification_confidence=mean_conf,
            claude_fallback_count=claude_fallbacks,
            page_count=page_count,
            qwen_override_rate=qwen_override_rate,
        )
        result.escalation = escalate.should_escalate(report)
    except Exception as exc:
        result.stage_errors.append((8, f"escalate: {exc}"))

    return result
