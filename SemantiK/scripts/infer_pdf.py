"""Run the 8-stage pipeline end-to-end on a real PDF.

Loads the trained stage-3 classifier, runs a PDF through stages 1→8, and
prints a structured summary. Saves the result JSON for eyeballing with
--save.

Usage:
    python scripts/infer_pdf.py /path/to/doc.pdf
    python scripts/infer_pdf.py doc.pdf --save /tmp/result.json
    python scripts/infer_pdf.py doc.pdf --no-classifier  (rules only)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dart_semantic.classify import load_classifier
from dart_semantic.pipeline import run_pipeline
from dart_semantic.reason import load_reasoner, unload_reasoner
from dart_semantic.glm_ocr import load_glm_ocr, unload_glm_ocr


def summarize(result) -> None:
    print(f"\n=== PDF: {result.pdf_path} ===")
    if result.stage_errors:
        print(f"[stage errors]")
        for stage, err in result.stage_errors:
            print(f"  stage {stage}: {err}")

    print(f"[stages]")
    print(f"  raw blocks:        {len(result.raw_blocks)}")
    print(f"  feature blocks:    {len(result.feature_blocks)}")
    print(f"  classified blocks: {len(result.classified_blocks)}")
    print(f"  resolved blocks:   {len(result.resolved_blocks)}")
    print(f"  html chars:        {len(result.html) if result.html else 'n/a'}")

    if result.classified_blocks:
        role_counts = Counter(cb.role for cb in result.classified_blocks)
        print(f"\n[role distribution]")
        for role, n in role_counts.most_common(10):
            print(f"  {role:25} {n}")

        # Bucket sources into the categories that matter:
        #   qwen3:agree     Qwen confirmed DistilBERT's hint
        #   qwen3:override  Qwen disagreed with DistilBERT and overrode
        #   qwen3:miss      Qwen's chunk was unparseable / missed the block
        #                   (pipeline fell back to DistilBERT's label)
        #   model           DistilBERT classified, Qwen wasn't run
        #   rule:...        pre-DistilBERT rule fired (model-less mode)
        def _bucket(s: str) -> str:
            if s in ("qwen3:agree", "qwen3:override"):
                return s
            if s.endswith(":qwen_miss"):
                return "qwen3:miss"
            return s.split(":")[0]
        src_counts = Counter(_bucket(cb.source) for cb in result.classified_blocks)
        print(f"\n[classification source]")
        for src, n in src_counts.most_common():
            print(f"  {src:16} {n}")

    if result.validation:
        print(f"\n[axe wcag22aa]  ok={result.validation.ok}  "
              f"violations={len(result.validation.violations)}")
        for v in result.validation.violations[:5]:
            print(f"  - {v}")

    if result.escalation:
        print(f"\n[escalation]  verdict={result.escalation.verdict}")
        for r in result.escalation.reasons:
            print(f"  - {r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--classifier", type=Path,
                    default=Path("models/classifier_v5/final"),
                    help="Path to the trained stage-3a classifier directory. "
                         "Default tracks the classifier version used to build "
                         "the current reasoner's training data — see "
                         "scripts/run_v7_full_rebuild.sh.")
    ap.add_argument("--no-classifier", action="store_true",
                    help="Run pipeline with rules only (ambiguous blocks default to paragraph)")
    ap.add_argument("--reasoner", action="store_true",
                    help="Also run stage-3b Qwen 3 reasoner as final labeler")
    ap.add_argument("--reasoner-model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="HF id for the stage-3b base model")
    ap.add_argument("--reasoner-adapter", type=Path, default=None,
                    help="Path to a fine-tuned LoRA adapter (e.g. models/reasoner/final)")
    ap.add_argument("--glm-ocr", action="store_true",
                    help="Enrich stage-1 extract with GLM-OCR on every detected "
                         "math/table region plus full-page fallback for any "
                         "page that had to use Tesseract. Results are cached "
                         "under data/glm_ocr_cache/.")
    ap.add_argument("--glm-ocr-model", default="zai-org/GLM-OCR",
                    help="HF id for the GLM-OCR model.")
    ap.add_argument("--language", default="en",
                    help="BCP-47 document language (html lang= attribute)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Write the full result (input+output+intermediates) as JSON")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        sys.exit(1)

    classifier = None
    if not args.no_classifier:
        if not args.classifier.exists():
            print(f"error: classifier {args.classifier} not found. "
                  f"Run `python train_classifier.py` first, or use --no-classifier.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[load] classifier {args.classifier}", file=sys.stderr)
        classifier = load_classifier(args.classifier)

    reasoner = None
    if args.reasoner:
        if args.reasoner_adapter:
            print(f"[load] reasoner {args.reasoner_model} + adapter "
                  f"{args.reasoner_adapter}", file=sys.stderr)
            reasoner = load_reasoner(args.reasoner_model,
                                     adapter_path=args.reasoner_adapter)
        else:
            print(f"[load] reasoner {args.reasoner_model} (zero-shot)",
                  file=sys.stderr)
            reasoner = load_reasoner(args.reasoner_model)

    glm_ocr = None
    if args.glm_ocr:
        print(f"[load] glm-ocr {args.glm_ocr_model}", file=sys.stderr)
        glm_ocr = load_glm_ocr(args.glm_ocr_model)

    try:
        print(f"[run] {args.pdf.name}", file=sys.stderr)
        result = run_pipeline(args.pdf, language=args.language,
                              classifier=classifier, reasoner=reasoner,
                              glm_ocr=glm_ocr)
    finally:
        if reasoner is not None:
            unload_reasoner(reasoner)
        if glm_ocr is not None:
            unload_glm_ocr(glm_ocr)

    summarize(result)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Pull out Qwen override diagnostics so the llm_fallback consumer
        # (Claude Code in dev, frontier LLM in prod) has enough signal to
        # act without re-running the pipeline.
        qwen_flips = []
        for i, cb in enumerate(result.classified_blocks):
            if cb.source == "qwen3:override":
                qwen_flips.append({
                    "block_id": i,
                    "role": cb.role,
                    "text": cb.features.raw.text[:140],
                    "page": cb.features.raw.page,
                })
        n_qwen_agree = sum(1 for cb in result.classified_blocks
                           if cb.source == "qwen3:agree")
        n_qwen_override = len(qwen_flips)
        n_qwen_miss = sum(1 for cb in result.classified_blocks
                          if cb.source.endswith(":qwen_miss"))

        args.save.write_text(json.dumps({
            "pdf_path": str(result.pdf_path),
            "stage_errors": result.stage_errors,
            "html": result.html,
            "n_raw_blocks": len(result.raw_blocks),
            "n_classified_blocks": len(result.classified_blocks),
            "role_counts": dict(Counter(cb.role for cb in result.classified_blocks)),
            "qwen": {
                "n_agree": n_qwen_agree,
                "n_override": n_qwen_override,
                "n_miss": n_qwen_miss,
                "override_samples": qwen_flips[:50],
            },
            "validation_ok": result.validation.ok if result.validation else None,
            "axe_violations": ([str(v) for v in result.validation.violations]
                               if result.validation else None),
            "escalation_verdict": (result.escalation.verdict
                                   if result.escalation else None),
            "escalation_reasons": (result.escalation.reasons
                                   if result.escalation else []),
        }, indent=2))
        print(f"\n[write] {args.save}", file=sys.stderr)
        if result.escalation and result.escalation.verdict == "llm_fallback":
            print(f"[fallback] verdict=llm_fallback — hand {args.save} to "
                  f"the fallback LLM (Claude Code in dev)", file=sys.stderr)


if __name__ == "__main__":
    main()
